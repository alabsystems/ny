// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Relaxed Clipping: Shrink input domains using CROWN linear constraints.
//!
//! This implements the Relaxed Clipping algorithm from Clip-and-Verify, which
//! uses a closed-form 1D solution to tighten input bounds constraint-by-constraint.
//!
//! ## Algorithm Overview
//!
//! Given input box `[x_L, x_U]` and CROWN linear constraints `{A_k·x + b_k ≤ 0}`,
//! find tighter bounds `[x_L', x_U']` that exclude infeasible regions.
//!
//! For each constraint and each input dimension, we solve for the extremal x value
//! that satisfies the constraint.
//!
//! ## References
//!
//! - Design: `designs/2026-01-28-clip-and-verify-algorithms.md` Section 2
//! - Baseline: `alpha-beta-CROWN/complete_verifier/input_split/clip.py`
//! - Paper: Wei et al., "Clip and Verify: Fast and Accurate Neural Network
//!   Verification via Clipping," arXiv:2512.11087

use ndarray::{Array2, ArrayD};
use ny_core::{
    dd::{next_down_f64, next_up_f64},
    Result,
};
use ny_core::{nan_propagating_max, nan_propagating_min};
use ny_tensor::{next_down_f32, next_up_f32};

use crate::bounds::{certified_affine_sum_f32, OutwardDirection};

struct RelaxedClipOptions {
    num_iterations: usize,
    is_lower: bool,
    preserve_infeasible: bool,
}

/// Outward enclosure of `a·x_hat ± |a|·eps + bias`.
///
/// The shared self-checked DD reducer keeps cancellation-heavy constraints
/// tight while retaining a directed-per-add fallback for non-finite inputs.
fn affine_center_radius_outward<F>(len: usize, bias: f32, is_lower: bool, mut term_at: F) -> f64
where
    F: FnMut(usize) -> (f32, f32, f32),
{
    affine_center_radius_except_outward(len, None, bias, is_lower, &mut term_at)
}

fn affine_center_radius_except_outward<F>(
    len: usize,
    skip: Option<usize>,
    bias: f32,
    is_lower: bool,
    mut term_at: F,
) -> f64
where
    F: FnMut(usize) -> (f32, f32, f32),
{
    let direction = if is_lower {
        OutwardDirection::Lower
    } else {
        OutwardDirection::Upper
    };
    certified_affine_sum_f32(
        bias,
        (0..len)
            .filter(|index| Some(*index) != skip)
            .flat_map(|index| {
                let (coefficient, midpoint, epsilon) = term_at(index);
                let radius_coefficient = if is_lower {
                    -coefficient.abs()
                } else {
                    coefficient.abs()
                };
                [(coefficient, midpoint), (radius_coefficient, epsilon)]
            }),
        direction,
    )
}

#[inline]
fn outward_clip_candidate(
    threshold: f32,
    concrete_without_dimension: f64,
    coefficient: f32,
    is_lower_form: bool,
) -> f32 {
    let numerator = if is_lower_form {
        next_up_f64(f64::from(threshold) - concrete_without_dimension)
    } else {
        next_down_f64(f64::from(threshold) - concrete_without_dimension)
    };
    let quotient = numerator / f64::from(coefficient);
    if coefficient < 0.0 {
        next_down_f32(next_down_f64(quotient) as f32)
    } else {
        next_up_f32(next_up_f64(quotient) as f32)
    }
}

/// Relaxed Clipping: shrink input domain using linear constraints.
///
/// This is the CPU reference implementation for testing and verification.
/// GPU acceleration follows via `ny-gpu` constraint buffers.
///
/// # Arguments
///
/// * `x_l` - Lower bounds, shape: `(batch, x_dim)`
/// * `x_u` - Upper bounds, shape: `(batch, x_dim)`
/// * `l_a` - CROWN coefficients, shape: `(batch, n_spec, x_dim)`
/// * `lbias` - CROWN bias, shape: `(batch, n_spec)`
/// * `thresholds` - Verification thresholds (usually zeros), shape: `(batch, n_spec)`
/// * `num_iterations` - Number of iterative refinement passes (typically 1)
/// * `is_lower` - True for lower bound problem (dm_lb), false for upper
///
/// # Returns
///
/// Tuple of tightened `(x_l_new, x_u_new)` with same shapes as input.
///
/// # Example
///
/// ```text
/// use ny_propagate::relaxed_clip;
/// use ndarray::{array, Array2, Array3};
///
/// // Single batch, single spec, 2D input
/// let x_l = array![[0.0, 0.0]];  // shape (1, 2)
/// let x_u = array![[1.0, 1.0]];  // shape (1, 2)
/// let l_a = array![[[1.0, 1.0]]]; // shape (1, 1, 2): x1 + x2
/// let lbias = array![[-0.5]];    // shape (1, 1): offset
/// let thresholds = array![[0.0]]; // shape (1, 1)
///
/// let (new_l, new_u) = relaxed_clip(
///     &x_l.into_dyn(), &x_u.into_dyn(),
///     &l_a.into_dyn(), &lbias.into_dyn(), &thresholds.into_dyn(),
///     1, true,
/// ).unwrap();
/// ```
///
/// # References
///
/// - `designs/2026-01-28-clip-and-verify-algorithms.md:188`
/// - `alpha-beta-CROWN/complete_verifier/input_split/clip.py:_clip_main_fn`
pub fn relaxed_clip(
    x_l: &ArrayD<f32>,
    x_u: &ArrayD<f32>,
    l_a: &ArrayD<f32>,
    lbias: &ArrayD<f32>,
    thresholds: &ArrayD<f32>,
    num_iterations: usize,
    is_lower: bool,
) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
    let (new_l, new_u, _) = relaxed_clip_internal(
        x_l,
        x_u,
        l_a,
        lbias,
        thresholds,
        RelaxedClipOptions {
            num_iterations,
            is_lower,
            preserve_infeasible: false,
        },
    )?;
    Ok((new_l, new_u))
}

/// Relaxed clipping variant that reports when multi-spec clipping makes a domain infeasible.
///
/// When `verified_by_clip[b]` is true, the returned bounds for batch `b` may have
/// `x_l > x_u`; callers must inspect the mask before constructing a `BoundedTensor`.
pub(crate) fn relaxed_clip_with_infeasible_mask(
    x_l: &ArrayD<f32>,
    x_u: &ArrayD<f32>,
    l_a: &ArrayD<f32>,
    lbias: &ArrayD<f32>,
    thresholds: &ArrayD<f32>,
    num_iterations: usize,
    is_lower: bool,
) -> Result<(ArrayD<f32>, ArrayD<f32>, Vec<bool>)> {
    relaxed_clip_internal(
        x_l,
        x_u,
        l_a,
        lbias,
        thresholds,
        RelaxedClipOptions {
            num_iterations,
            is_lower,
            preserve_infeasible: true,
        },
    )
}

/// #lsnc-relaxed-clip-fast gate. The relaxed input-clip loop is the dominant
/// SERIAL cost of the graph input-split BaB on small nets (MEASURED on real
/// lsnc_relu: `push_batched_clip_children -> relaxed_clip_internal` = ~54% of
/// the main-thread critical path, and a RAYON_NUM_THREADS 1/4/16 sweep showed
/// domains/s is FLAT — the wall is serial, not thread-sync). The historical
/// `relaxed_clip_internal_scalar` walks the batch with per-element
/// `ndarray::Index` (`[[b, s]]` / `[[b, dim]]`) and re-allocates a
/// `(batch, n_spec)` column via `extract_dim_slice` PLUS `x_hat_dim`/`eps_dim`
/// `Vec`s once PER dimension PER iteration — pure indexing/allocator overhead.
/// `relaxed_clip_internal_fast` computes the SAME arithmetic in the SAME order
/// over flat row-major slices with no per-dim allocation, so it is
/// BIT-IDENTICAL to the scalar reference (see `test_relaxed_clip_fast_scalar_parity`).
/// Default ON; set `NY_RELAXED_CLIP_FAST=0` to force the scalar reference (the
/// A/B + parity baseline), mirroring the `NY_INPUT_SPLIT_*` gates.
fn relaxed_clip_fast_enabled() -> bool {
    !matches!(
        std::env::var("NY_RELAXED_CLIP_FAST").ok().as_deref(),
        Some("0") | Some("false")
    )
}

fn relaxed_clip_internal(
    x_l: &ArrayD<f32>,
    x_u: &ArrayD<f32>,
    l_a: &ArrayD<f32>,
    lbias: &ArrayD<f32>,
    thresholds: &ArrayD<f32>,
    options: RelaxedClipOptions,
) -> Result<(ArrayD<f32>, ArrayD<f32>, Vec<bool>)> {
    if relaxed_clip_fast_enabled() {
        relaxed_clip_internal_fast(x_l, x_u, l_a, lbias, thresholds, options)
    } else {
        relaxed_clip_internal_scalar(x_l, x_u, l_a, lbias, thresholds, options)
    }
}

fn relaxed_clip_internal_scalar(
    x_l: &ArrayD<f32>,
    x_u: &ArrayD<f32>,
    l_a: &ArrayD<f32>,
    lbias: &ArrayD<f32>,
    thresholds: &ArrayD<f32>,
    options: RelaxedClipOptions,
) -> Result<(ArrayD<f32>, ArrayD<f32>, Vec<bool>)> {
    // Validate shapes
    let x_shape = x_l.shape();
    let x_u_shape = x_u.shape();
    let l_a_shape = l_a.shape();

    if x_shape.len() != 2 {
        return Err(ny_core::NyError::InvalidSpec(format!(
            "x_l must be 2D (batch, x_dim), got {:?}",
            x_shape
        )));
    }
    if x_u_shape.len() != 2 {
        return Err(ny_core::NyError::InvalidSpec(format!(
            "x_u must be 2D (batch, x_dim), got {:?}",
            x_u_shape
        )));
    }
    if l_a_shape.len() != 3 {
        return Err(ny_core::NyError::InvalidSpec(format!(
            "l_a must be 3D (batch, n_spec, x_dim), got {:?}",
            l_a_shape
        )));
    }

    let batch = x_shape[0];
    let x_dim = x_shape[1];
    let n_spec = l_a_shape[1];

    if l_a_shape[0] != batch || l_a_shape[2] != x_dim {
        return Err(ny_core::NyError::shape_mismatch(
            vec![batch, n_spec, x_dim],
            l_a_shape.to_vec(),
        ));
    }
    if x_u_shape[0] != batch || x_u_shape[1] != x_dim {
        return Err(ny_core::NyError::shape_mismatch(
            vec![batch, x_dim],
            x_u_shape.to_vec(),
        ));
    }
    if lbias.ndim() != 2 || lbias.shape()[0] != batch || lbias.shape()[1] != n_spec {
        return Err(ny_core::NyError::shape_mismatch(
            vec![batch, n_spec],
            lbias.shape().to_vec(),
        ));
    }
    if thresholds.ndim() != 2 || thresholds.shape()[0] != batch || thresholds.shape()[1] != n_spec {
        return Err(ny_core::NyError::shape_mismatch(
            vec![batch, n_spec],
            thresholds.shape().to_vec(),
        ));
    }

    let mut x_l_out = x_l.clone();
    let mut x_u_out = x_u.clone();
    let mut verified_by_clip = vec![false; batch];

    for _iter in 0..options.num_iterations {
        // clip.py forms xhat=(x_U+x_L)/2 and eps=(x_U-x_L)/2, which only
        // describe a valid box while the batch is still active.
        for b in 0..batch {
            if verified_by_clip[b] {
                continue;
            }
            for dim in 0..x_dim {
                if x_l_out[[b, dim]] > x_u_out[[b, dim]] {
                    return Err(ny_core::NyError::InvalidSpec(format!(
                        "relaxed_clip: x_l > x_u at batch={} dim={}",
                        b, dim
                    )));
                }
            }
        }

        // x_hat = (x_L + x_U) / 2  (centroid)
        // eps = (x_U - x_L) / 2    (half-widths)
        let x_hat = (&x_l_out + &x_u_out) / 2.0;
        let eps = (&x_u_out - &x_l_out) / 2.0;

        // For each dimension, compute the clipping update
        // We iterate per-dimension to avoid creating large intermediate tensors
        for dim in 0..x_dim {
            // Extract l_a column for this dimension: shape (batch, n_spec)
            let l_a_dim = extract_dim_slice(l_a, dim);

            // Solve for the clip candidate x_i* = (threshold - concrete_minus_one) / l_a_dim,
            // where concrete_minus_one is `dm_lb` with the contribution of dimension `dim`
            // removed (the min contribution of all *other* dimensions plus bias).
            //
            // Directed-OUTWARD rounding (soundness, task #19): the clipped box must contain
            // EVERY point where the spec can still be violated, so a lower candidate
            // (l_a_dim < 0) must round DOWN toward -inf and an upper candidate
            // (l_a_dim > 0) must round UP toward +inf. Round-to-nearest here could round a
            // candidate INWARD by up to half a ULP and cut a boundary violation — a false
            // "hold". Both bound directions want `concrete_minus_one` minimized (it enters
            // the numerator negated), so we keep it an underestimate: `dm_lb` is already a
            // round-DOWN concretization, and the per-dim reconstruction is done in f64 so no
            // extra round-to-nearest error creeps in before the single directed round. This
            // mirrors the directed rounding in `concretize_bounds` (#2303).
            // Shape: (batch, n_spec)
            let mut curr_x = Array2::<f32>::zeros((batch, n_spec));
            for b in 0..batch {
                for s in 0..n_spec {
                    let a_val = l_a_dim[[b, s]];
                    if a_val.abs() > 1e-10 {
                        let concrete_minus_one = affine_center_radius_except_outward(
                            x_dim,
                            Some(dim),
                            lbias[[b, s]],
                            options.is_lower,
                            |other_dim| {
                                (
                                    l_a[[b, s, other_dim]],
                                    x_hat[[b, other_dim]],
                                    eps[[b, other_dim]],
                                )
                            },
                        );
                        curr_x[[b, s]] = outward_clip_candidate(
                            thresholds[[b, s]],
                            concrete_minus_one,
                            a_val,
                            options.is_lower,
                        );
                    } else {
                        // Coefficient near zero: no constraint on this dimension
                        curr_x[[b, s]] = if a_val < 0.0 {
                            f32::NEG_INFINITY
                        } else {
                            f32::INFINITY
                        };
                    }
                }
            }

            // Candidate selection based on sign of coefficient
            // - l_a < 0 gives candidates for x_L (lower bound increases)
            // - l_a > 0 gives candidates for x_U (upper bound decreases)
            for b in 0..batch {
                if verified_by_clip[b] {
                    continue;
                }

                // Find max over specs for lower bound candidates (where l_a < 0)
                let mut max_lower_candidate = f32::NEG_INFINITY;
                for s in 0..n_spec {
                    if l_a_dim[[b, s]] < 0.0 {
                        // NaN-propagating: if curr_x is NaN (from division by
                        // near-zero coefficient), propagate rather than absorb
                        // via IEEE 754 max semantics (#2812 Slice 1).
                        max_lower_candidate =
                            nan_propagating_max(max_lower_candidate, curr_x[[b, s]]);
                    }
                }

                // Find min over specs for upper bound candidates (where l_a > 0)
                let mut min_upper_candidate = f32::INFINITY;
                for s in 0..n_spec {
                    if l_a_dim[[b, s]] > 0.0 {
                        min_upper_candidate =
                            nan_propagating_min(min_upper_candidate, curr_x[[b, s]]);
                    }
                }

                // Update bounds (clamp to original bounds).
                // NaN-propagating: if candidate is NaN, the updated bound
                // becomes NaN rather than silently keeping the old value,
                // so downstream NaN firewalls can detect the corruption.
                let new_lower = nan_propagating_max(x_l_out[[b, dim]], max_lower_candidate);
                let new_upper = nan_propagating_min(x_u_out[[b, dim]], min_upper_candidate);

                // Ensure valid bounds (lower <= upper).
                // NaN bounds are also invalid — NaN comparisons return false
                // in IEEE 754, so check explicitly.
                if new_lower.is_nan() || new_upper.is_nan() {
                    x_l_out[[b, dim]] = x_l[[b, dim]];
                    x_u_out[[b, dim]] = x_u[[b, dim]];
                    continue;
                }
                if new_lower > new_upper {
                    if options.preserve_infeasible {
                        x_l_out[[b, dim]] = new_lower;
                        x_u_out[[b, dim]] = new_upper;
                        verified_by_clip[b] = true;
                    } else {
                        // Clipping made bounds invalid, revert to original.
                        x_l_out[[b, dim]] = x_l[[b, dim]];
                        x_u_out[[b, dim]] = x_u[[b, dim]];
                    }
                    continue;
                }

                x_l_out[[b, dim]] = new_lower;
                x_u_out[[b, dim]] = new_upper;
            }
        }
    }

    Ok((x_l_out, x_u_out, verified_by_clip))
}

/// Flat-slice, allocation-free reimplementation of [`relaxed_clip_internal_scalar`].
///
/// BIT-IDENTICAL to the scalar reference: every floating-point operation is the
/// same expression evaluated in the same order (per-dimension f64 reconstruction,
/// directed `next_down_f32`/`next_up_f32` rounding, NaN-propagating max/min, and
/// revert-to-ORIGINAL-input on NaN / infeasible-without-`preserve_infeasible`).
/// The only change is mechanical: read/write via row-major flat indices on
/// standard-layout slices instead of `ndarray::Index`, and hoist the per-dim
/// `extract_dim_slice` / `x_hat_dim` / `eps_dim` allocations out entirely. The
/// per-`(b, dim)` update only reads iteration-start snapshots (`x_hat`, `eps`,
/// `dm_lb`) and the current `xl[b,dim]`/`xu[b,dim]` (untouched by other dims this
/// iteration), and dims are still processed in order 0..x_dim with the same
/// `verified`-skip, so the observable result is unchanged. Guarded by
/// `test_relaxed_clip_fast_scalar_parity`.
#[allow(clippy::needless_range_loop)]
fn relaxed_clip_internal_fast(
    x_l: &ArrayD<f32>,
    x_u: &ArrayD<f32>,
    l_a: &ArrayD<f32>,
    lbias: &ArrayD<f32>,
    thresholds: &ArrayD<f32>,
    options: RelaxedClipOptions,
) -> Result<(ArrayD<f32>, ArrayD<f32>, Vec<bool>)> {
    // Validate shapes (identical checks to the scalar reference).
    let x_shape = x_l.shape();
    let x_u_shape = x_u.shape();
    let l_a_shape = l_a.shape();

    if x_shape.len() != 2 {
        return Err(ny_core::NyError::InvalidSpec(format!(
            "x_l must be 2D (batch, x_dim), got {:?}",
            x_shape
        )));
    }
    if x_u_shape.len() != 2 {
        return Err(ny_core::NyError::InvalidSpec(format!(
            "x_u must be 2D (batch, x_dim), got {:?}",
            x_u_shape
        )));
    }
    if l_a_shape.len() != 3 {
        return Err(ny_core::NyError::InvalidSpec(format!(
            "l_a must be 3D (batch, n_spec, x_dim), got {:?}",
            l_a_shape
        )));
    }

    let batch = x_shape[0];
    let x_dim = x_shape[1];
    let n_spec = l_a_shape[1];

    if l_a_shape[0] != batch || l_a_shape[2] != x_dim {
        return Err(ny_core::NyError::shape_mismatch(
            vec![batch, n_spec, x_dim],
            l_a_shape.to_vec(),
        ));
    }
    if x_u_shape[0] != batch || x_u_shape[1] != x_dim {
        return Err(ny_core::NyError::shape_mismatch(
            vec![batch, x_dim],
            x_u_shape.to_vec(),
        ));
    }
    if lbias.ndim() != 2 || lbias.shape()[0] != batch || lbias.shape()[1] != n_spec {
        return Err(ny_core::NyError::shape_mismatch(
            vec![batch, n_spec],
            lbias.shape().to_vec(),
        ));
    }
    if thresholds.ndim() != 2 || thresholds.shape()[0] != batch || thresholds.shape()[1] != n_spec {
        return Err(ny_core::NyError::shape_mismatch(
            vec![batch, n_spec],
            thresholds.shape().to_vec(),
        ));
    }

    let is_lower = options.is_lower;

    // Standard-layout (row-major, contiguous) views so `.as_slice()` yields flat
    // C-order slices with the same logical `[[..]]` element order.
    let l_a_std = l_a.as_standard_layout();
    let l_a_s = l_a_std.as_slice().expect("l_a standard layout contiguous");
    let lbias_std = lbias.as_standard_layout();
    let lbias_s = lbias_std
        .as_slice()
        .expect("lbias standard layout contiguous");
    let thr_std = thresholds.as_standard_layout();
    let thr_s = thr_std
        .as_slice()
        .expect("thresholds standard layout contiguous");
    let x_l_std = x_l.as_standard_layout();
    let orig_l = x_l_std.as_slice().expect("x_l standard layout contiguous");
    let x_u_std = x_u.as_standard_layout();
    let orig_u = x_u_std.as_slice().expect("x_u standard layout contiguous");

    // Working bounds (batch * x_dim, row-major). Reverts restore ORIGINAL inputs.
    let mut xl = orig_l.to_vec();
    let mut xu = orig_u.to_vec();
    let mut verified_by_clip = vec![false; batch];

    // Reused scratch (allocated once, not per dim / per iteration).
    let mut x_hat = vec![0f32; batch * x_dim];
    let mut eps = vec![0f32; batch * x_dim];
    let mut curr_x = vec![0f32; batch * n_spec];

    for _iter in 0..options.num_iterations {
        // Validity check (skip already-verified batches), matching the scalar path.
        for b in 0..batch {
            if verified_by_clip[b] {
                continue;
            }
            for dim in 0..x_dim {
                let idx = b * x_dim + dim;
                if xl[idx] > xu[idx] {
                    return Err(ny_core::NyError::InvalidSpec(format!(
                        "relaxed_clip: x_l > x_u at batch={} dim={}",
                        b, dim
                    )));
                }
            }
        }

        // x_hat = (x_L + x_U) / 2 ; eps = (x_U - x_L) / 2 (over the whole batch).
        // Bit-identical centroid anchor: f32::midpoint rounds differently at
        // overflow/subnormal edges and the produced bounds must not move.
        #[allow(clippy::manual_midpoint)]
        for i in 0..batch * x_dim {
            x_hat[i] = (xl[i] + xu[i]) / 2.0;
            eps[i] = (xu[i] - xl[i]) / 2.0;
        }

        for dim in 0..x_dim {
            // curr_x[b, s] for this dimension (computed for all b, exactly as scalar).
            for b in 0..batch {
                let a_base = b * n_spec * x_dim + dim;
                let s_base = b * n_spec;
                for s in 0..n_spec {
                    let a_val = l_a_s[a_base + s * x_dim];
                    curr_x[s_base + s] = if a_val.abs() > 1e-10 {
                        let concrete_minus_one = affine_center_radius_except_outward(
                            x_dim,
                            Some(dim),
                            lbias_s[s_base + s],
                            is_lower,
                            |other_dim| {
                                (
                                    l_a_s[b * n_spec * x_dim + s * x_dim + other_dim],
                                    x_hat[b * x_dim + other_dim],
                                    eps[b * x_dim + other_dim],
                                )
                            },
                        );
                        outward_clip_candidate(
                            thr_s[s_base + s],
                            concrete_minus_one,
                            a_val,
                            is_lower,
                        )
                    } else if a_val < 0.0 {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    };
                }
            }

            // Candidate selection + bound update (skip verified batches).
            for b in 0..batch {
                if verified_by_clip[b] {
                    continue;
                }
                let a_base = b * n_spec * x_dim + dim;
                let s_base = b * n_spec;

                let mut max_lower_candidate = f32::NEG_INFINITY;
                for s in 0..n_spec {
                    if l_a_s[a_base + s * x_dim] < 0.0 {
                        max_lower_candidate =
                            nan_propagating_max(max_lower_candidate, curr_x[s_base + s]);
                    }
                }

                let mut min_upper_candidate = f32::INFINITY;
                for s in 0..n_spec {
                    if l_a_s[a_base + s * x_dim] > 0.0 {
                        min_upper_candidate =
                            nan_propagating_min(min_upper_candidate, curr_x[s_base + s]);
                    }
                }

                let idx = b * x_dim + dim;
                let new_lower = nan_propagating_max(xl[idx], max_lower_candidate);
                let new_upper = nan_propagating_min(xu[idx], min_upper_candidate);

                if new_lower.is_nan() || new_upper.is_nan() {
                    xl[idx] = orig_l[idx];
                    xu[idx] = orig_u[idx];
                    continue;
                }
                if new_lower > new_upper {
                    if options.preserve_infeasible {
                        xl[idx] = new_lower;
                        xu[idx] = new_upper;
                        verified_by_clip[b] = true;
                    } else {
                        xl[idx] = orig_l[idx];
                        xu[idx] = orig_u[idx];
                    }
                    continue;
                }

                xl[idx] = new_lower;
                xu[idx] = new_upper;
            }
        }
    }

    let x_l_out = ArrayD::from_shape_vec(vec![batch, x_dim], xl)
        .map_err(|e| ny_core::NyError::InvalidSpec(format!("relaxed_clip fast x_l_out: {}", e)))?;
    let x_u_out = ArrayD::from_shape_vec(vec![batch, x_dim], xu)
        .map_err(|e| ny_core::NyError::InvalidSpec(format!("relaxed_clip fast x_u_out: {}", e)))?;
    Ok((x_l_out, x_u_out, verified_by_clip))
}

/// #lsnc-clip-planes (S5): caller-owned scratch for
/// [`relaxed_clip_single_spec_row_fast`], reused across the sequential
/// threshold rows of one child batch so the per-row clip does zero heap
/// allocations (the historical path allocated ~8 batch-sized buffers per
/// threshold row inside `relaxed_clip_internal_fast`).
pub(crate) struct SingleSpecRowClipScratch {
    /// Row-call original bounds — the NaN / infeasible revert target (I-A5).
    /// Snapshot of the working box at row entry, mirroring the `x_l`/`x_u`
    /// arguments of the per-row `relaxed_clip_with_infeasible_mask` call.
    row_orig_l: Vec<f32>,
    row_orig_u: Vec<f32>,
    /// Per-entry converged flag: a full iteration pass that changed no bit of
    /// the entry's box and latched nothing is a fixed point — every later
    /// iteration recomputes identical values — so the entry is elided from
    /// further iterations (pure elision, bit-identical outputs).
    stable: Vec<bool>,
    /// Per-entry infeasibility latch of the CURRENT row call, fresh per row
    /// (mirrors the fresh `verified_by_clip` of each per-row
    /// `relaxed_clip_with_infeasible_mask` call; the caller merges it into its
    /// cross-row latch).
    pub(crate) row_verified: Vec<bool>,
    /// Iteration-start centroid / half-width snapshots for ONE entry.
    x_hat: Vec<f32>,
    eps: Vec<f32>,
}

impl SingleSpecRowClipScratch {
    pub(crate) fn new() -> Self {
        Self {
            row_orig_l: Vec::new(),
            row_orig_u: Vec::new(),
            stable: Vec::new(),
            row_verified: Vec::new(),
            x_hat: Vec::new(),
            eps: Vec::new(),
        }
    }

    fn reset(&mut self, x_dim: usize, xl: &[f32], xu: &[f32]) {
        self.row_orig_l.clear();
        self.row_orig_l.extend_from_slice(xl);
        self.row_orig_u.clear();
        self.row_orig_u.extend_from_slice(xu);
        let m = xl.len().checked_div(x_dim).unwrap_or(0);
        self.stable.clear();
        self.stable.resize(m, false);
        self.row_verified.clear();
        self.row_verified.resize(m, false);
        self.x_hat.clear();
        self.x_hat.resize(x_dim, 0.0);
        self.eps.clear();
        self.eps.resize(x_dim, 0.0);
    }
}

/// #lsnc-clip-planes (S5): ONE threshold row of the batched sequential relaxed
/// clip — the `n_spec = 1` specialization of [`relaxed_clip_internal_fast`]
/// with `preserve_infeasible = true`, caller-owned scratch, and in-place
/// working bounds.
///
/// BIT-PARITY CLASS (same expressions, same order — I-A1..I-A7): for every
/// batch entry `b` this evaluates exactly the arithmetic of
/// `relaxed_clip_internal_fast` on inputs `x_l = xl`, `x_u = xu`,
/// `l_a[b, 0, :] = a[b*x_dim..]`, `lbias[b, 0] = bias[b]`,
/// `thresholds[b, 0] = thr[b]`: the per-iteration centroid/half-width
/// snapshots, the in-order f64 `dm_lb` accumulation with directed rounding,
/// the f64 candidate reconstruction with a single directed outward round, the
/// near-zero `±inf` sentinel, the literal `nan_propagating_max/min` folds, the
/// NaN revert-to-row-original, and the infeasible latch-and-skip. Two
/// mechanical restructurings, both provably unobservable because batch entries
/// never interact in the reference (every quantity of entry `b` is a function
/// of entry `b`'s own state):
///
/// 1. loops are entry-major (all dims of `b` before `b+1`) instead of
///    dim-major — every read still sees the same value it did in the
///    reference's order;
/// 2. converged entries are elided (`stable`): an iteration pass that leaves
///    an entry's `(xl, xu, latch)` bit-unchanged is a fixed point, so all
///    remaining iterations are no-ops for that entry, and the whole row exits
///    early when every entry is verified or stable.
///
/// On success, `xl`/`xu` hold the row-clipped bounds and
/// `scratch.row_verified` the per-entry infeasibility latch of this row.
/// Parity: `test_relaxed_clip_single_spec_row_parity` (direct, vs the fast
/// reference across option/NaN/±inf/near-zero/infeasible fixtures) and
/// `test_batched_clip_planes_matches_stacked_s5` (through the sequential
/// threshold loop).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub(crate) fn relaxed_clip_single_spec_row_fast(
    xl: &mut [f32],
    xu: &mut [f32],
    a: &[f32],
    bias: &[f32],
    thr: &[f32],
    m: usize,
    x_dim: usize,
    num_iterations: usize,
    is_lower: bool,
    scratch: &mut SingleSpecRowClipScratch,
) -> Result<()> {
    if xl.len() != m * x_dim
        || xu.len() != m * x_dim
        || a.len() != m * x_dim
        || bias.len() != m
        || thr.len() != m
    {
        return Err(ny_core::NyError::InvalidSpec(format!(
            "relaxed_clip_single_spec_row_fast: shape mismatch (m={}, x_dim={}, xl={}, xu={}, a={}, bias={}, thr={})",
            m,
            x_dim,
            xl.len(),
            xu.len(),
            a.len(),
            bias.len(),
            thr.len()
        )));
    }
    scratch.reset(x_dim, xl, xu);

    let mut active = m;

    for _iter in 0..num_iterations {
        if active == 0 {
            break;
        }
        for b in 0..m {
            if scratch.row_verified[b] || scratch.stable[b] {
                continue;
            }
            let base = b * x_dim;

            // Validity check (identical error to the reference; skipped
            // entries passed it with the same box on their last active pass).
            for dim in 0..x_dim {
                if xl[base + dim] > xu[base + dim] {
                    return Err(ny_core::NyError::InvalidSpec(format!(
                        "relaxed_clip: x_l > x_u at batch={} dim={}",
                        b, dim
                    )));
                }
            }

            // Iteration-start snapshots for entry b (the reference computes
            // these for the whole batch before any update of the iteration;
            // entry b's row is untouched between that point and b's updates).
            // Bit-identical centroid anchor (BIT-PARITY class): f32::midpoint
            // rounds differently at overflow/subnormal edges and the produced
            // bounds must not move.
            #[allow(clippy::manual_midpoint)]
            for d in 0..x_dim {
                scratch.x_hat[d] = (xl[base + d] + xu[base + d]) / 2.0;
                scratch.eps[d] = (xu[base + d] - xl[base + d]) / 2.0;
            }

            let mut changed = false;
            for dim in 0..x_dim {
                let a_val = a[base + dim];
                let curr = if a_val.abs() > 1e-10 {
                    let concrete_minus_one = affine_center_radius_except_outward(
                        x_dim,
                        Some(dim),
                        bias[b],
                        is_lower,
                        |other_dim| {
                            (
                                a[base + other_dim],
                                scratch.x_hat[other_dim],
                                scratch.eps[other_dim],
                            )
                        },
                    );
                    outward_clip_candidate(thr[b], concrete_minus_one, a_val, is_lower)
                } else if a_val < 0.0 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                };

                // Single-spec candidate folds, evaluated literally through the
                // NaN-propagating helpers so every bit (incl. -0.0 / NaN)
                // matches the reference's spec-loop folds.
                let mut max_lower_candidate = f32::NEG_INFINITY;
                if a_val < 0.0 {
                    max_lower_candidate = nan_propagating_max(max_lower_candidate, curr);
                }
                let mut min_upper_candidate = f32::INFINITY;
                if a_val > 0.0 {
                    min_upper_candidate = nan_propagating_min(min_upper_candidate, curr);
                }

                let idx = base + dim;
                let new_lower = nan_propagating_max(xl[idx], max_lower_candidate);
                let new_upper = nan_propagating_min(xu[idx], min_upper_candidate);

                if new_lower.is_nan() || new_upper.is_nan() {
                    changed |= xl[idx].to_bits() != scratch.row_orig_l[idx].to_bits()
                        || xu[idx].to_bits() != scratch.row_orig_u[idx].to_bits();
                    xl[idx] = scratch.row_orig_l[idx];
                    xu[idx] = scratch.row_orig_u[idx];
                    continue;
                }
                if new_lower > new_upper {
                    // preserve_infeasible = true in this lane: store the
                    // inverted pair and latch; later dims are skipped exactly
                    // as the reference's verified-skip does.
                    xl[idx] = new_lower;
                    xu[idx] = new_upper;
                    scratch.row_verified[b] = true;
                    active -= 1;
                    break;
                }

                changed |= xl[idx].to_bits() != new_lower.to_bits()
                    || xu[idx].to_bits() != new_upper.to_bits();
                xl[idx] = new_lower;
                xu[idx] = new_upper;
            }

            if !scratch.row_verified[b] && !changed {
                scratch.stable[b] = true;
                active -= 1;
            }
        }
    }

    Ok(())
}

/// Concretize bounds using Hölder's inequality for L-inf norm.
///
/// Computes: `l_a·xhat + sign*|l_a|·eps + lbias`
///
/// For lower bound (is_lower=true), sign=-1 gives a sound lower bound.
/// For upper bound (is_lower=false), sign=+1 gives a sound upper bound.
///
/// # Arguments
///
/// * `x_hat` - Centroid of input box, shape: `(batch, x_dim)`
/// * `eps` - Half-widths of input box, shape: `(batch, x_dim)`
/// * `l_a` - CROWN coefficients, shape: `(batch, n_spec, x_dim)`
/// * `lbias` - CROWN bias, shape: `(batch, n_spec)`
/// * `is_lower` - True for lower bound (sign=-1)
///
/// # Returns
///
/// Concretized bounds, shape: `(batch, n_spec)`
///
/// # References
///
/// - `designs/2026-01-28-clip-and-verify-algorithms.md:250`
/// - `alpha-beta-CROWN/complete_verifier/input_split/clip.py:concretize_bounds`
pub fn concretize_bounds(
    x_hat: &ArrayD<f32>,
    eps: &ArrayD<f32>,
    l_a: &ArrayD<f32>,
    lbias: &ArrayD<f32>,
    is_lower: bool,
) -> Array2<f32> {
    // Directed rounding: lower bounds round down, upper bounds round up (#2303).
    let round: fn(f64) -> f32 = if is_lower {
        |v| next_down_f32(v as f32)
    } else {
        |v| next_up_f32(v as f32)
    };

    let batch = x_hat.shape()[0];
    let x_dim = x_hat.shape()[1];
    let n_spec = l_a.shape()[1];

    let mut result = Array2::<f32>::zeros((batch, n_spec));

    for b in 0..batch {
        for s in 0..n_spec {
            let val = affine_center_radius_outward(x_dim, lbias[[b, s]], is_lower, |d| {
                (l_a[[b, s, d]], x_hat[[b, d]], eps[[b, d]])
            });
            // NaN guard: if NaN entered the accumulator (e.g., from NaN CROWN
            // coefficients via safe_add overflow), fall back to conservative
            // bounds matching concretize_f64_inner (#2963, #2577).
            result[[b, s]] = if val.is_nan() {
                if is_lower {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                round(val)
            };
        }
    }

    result
}

/// Extract a slice of l_a for a specific dimension.
/// Returns shape (batch, n_spec) from l_a shape (batch, n_spec, x_dim).
fn extract_dim_slice(l_a: &ArrayD<f32>, dim: usize) -> Array2<f32> {
    let batch = l_a.shape()[0];
    let n_spec = l_a.shape()[1];
    let mut result = Array2::<f32>::zeros((batch, n_spec));

    for b in 0..batch {
        for s in 0..n_spec {
            result[[b, s]] = l_a[[b, s, dim]];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concretize_bounds_is_outward_under_catastrophic_cancellation() {
        let large = 2.0_f32.powi(50);
        let x_hat =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 3]), vec![large, 1.0, large]).unwrap();
        let eps = ArrayD::zeros(ndarray::IxDyn(&[1, 3]));
        let coefficients =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 3]), vec![large, 1.0, -large]).unwrap();
        let bias = ArrayD::zeros(ndarray::IxDyn(&[1, 1]));

        let lower = concretize_bounds(&x_hat, &eps, &coefficients, &bias, true);
        let upper = concretize_bounds(&x_hat, &eps, &coefficients, &bias, false);
        assert!(lower[[0, 0]] <= 1.0);
        assert!(
            upper[[0, 0]] >= 1.0,
            "upper {} must enclose exact 2^100 + 1 - 2^100",
            upper[[0, 0]]
        );
    }
    use ndarray::array;

    /// The fast flat-slice relaxed-clip path must be BIT-IDENTICAL to the scalar
    /// reference on an lsnc-shaped input-split batch (batch × n_spec × x_dim), over
    /// every option combination and across the multi-iteration refinement — this is
    /// the certified input-clip path, so any divergence is a soundness bug. Compares
    /// raw `f32` bit patterns (so signed zeros / NaNs / ±inf all match exactly) plus
    /// the `verified_by_clip` mask. #lsnc-relaxed-clip-fast.
    #[ntest::timeout(30000)]
    #[test]
    fn test_relaxed_clip_fast_scalar_parity() {
        let batch = 12usize;
        let n_spec = 39usize; // lsnc quadrotor2d: 13 clauses × 3 rows
        let x_dim = 6usize; // lsnc quadrotor2d state dim

        // Deterministic pseudo-random fixture (LCG) with coefficients spanning
        // negative / positive / near-zero (< 1e-10) so both candidate branches,
        // the near-zero ±inf path, and infeasible (verified) domains are exercised.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) // in [0,1)
        };

        let mut x_l = ArrayD::<f32>::zeros(vec![batch, x_dim]);
        let mut x_u = ArrayD::<f32>::zeros(vec![batch, x_dim]);
        for b in 0..batch {
            for d in 0..x_dim {
                let c = (next() - 0.5) * 4.0;
                let w = next() * 2.0 + 0.05;
                x_l[[b, d]] = c - w;
                x_u[[b, d]] = c + w;
            }
        }
        let mut l_a = ArrayD::<f32>::zeros(vec![batch, n_spec, x_dim]);
        for b in 0..batch {
            for s in 0..n_spec {
                for d in 0..x_dim {
                    let r = next();
                    l_a[[b, s, d]] = if r < 0.12 {
                        0.0 // near-zero coefficient path
                    } else {
                        (next() - 0.5) * 3.0
                    };
                }
            }
        }
        let mut lbias = ArrayD::<f32>::zeros(vec![batch, n_spec]);
        let mut thresholds = ArrayD::<f32>::zeros(vec![batch, n_spec]);
        for b in 0..batch {
            for s in 0..n_spec {
                lbias[[b, s]] = (next() - 0.5) * 2.0;
                // Tight thresholds so some domains clip aggressively / go infeasible.
                thresholds[[b, s]] = (next() - 0.5) * 0.2;
            }
        }

        for &is_lower in &[true, false] {
            for &preserve_infeasible in &[false, true] {
                for &num_iterations in &[1usize, 20] {
                    let opts = || RelaxedClipOptions {
                        num_iterations,
                        is_lower,
                        preserve_infeasible,
                    };
                    let (sl, su, sv) =
                        relaxed_clip_internal_scalar(&x_l, &x_u, &l_a, &lbias, &thresholds, opts())
                            .expect("scalar path");
                    let (fl, fu, fv) =
                        relaxed_clip_internal_fast(&x_l, &x_u, &l_a, &lbias, &thresholds, opts())
                            .expect("fast path");

                    assert_eq!(sv, fv, "verified mask mismatch (is_lower={is_lower}, preserve={preserve_infeasible}, iters={num_iterations})");
                    let sl = sl.as_slice().unwrap();
                    let su = su.as_slice().unwrap();
                    let fl = fl.as_slice().unwrap();
                    let fu = fu.as_slice().unwrap();
                    for i in 0..sl.len() {
                        assert_eq!(
                            sl[i].to_bits(), fl[i].to_bits(),
                            "x_l bit mismatch at {i} (is_lower={is_lower}, preserve={preserve_infeasible}, iters={num_iterations}): scalar={} fast={}",
                            sl[i], fl[i]
                        );
                        assert_eq!(
                            su[i].to_bits(), fu[i].to_bits(),
                            "x_u bit mismatch at {i} (is_lower={is_lower}, preserve={preserve_infeasible}, iters={num_iterations}): scalar={} fast={}",
                            su[i], fu[i]
                        );
                    }
                }
            }
        }
    }

    /// #lsnc-clip-planes (S5): the caller-scratch `n_spec = 1` row core must be
    /// BIT-IDENTICAL to the fast reference (`relaxed_clip_with_infeasible_mask`,
    /// i.e. `preserve_infeasible = true`) — bounds compared as raw f32 bits and
    /// the row latch compared exactly — across is_lower × iteration counts on an
    /// adversarial single-spec fixture: mixed-sign / exact-zero / near-zero
    /// coefficients, NaN coefficient, NaN bias, ±inf bias, tight thresholds
    /// driving infeasible latches, a zero-width box, and an infinite-bound box
    /// that drives the NaN candidate revert (I-A5). The fixed-point elision
    /// (`stable`) must be unobservable at every iteration count.
    #[ntest::timeout(30000)]
    #[test]
    fn test_relaxed_clip_single_spec_row_parity() {
        let m = 14usize;
        let x_dim = 6usize;

        let mut state: u64 = 0xC0FFEE123456789;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32)
        };

        let mut xl = vec![0f32; m * x_dim];
        let mut xu = vec![0f32; m * x_dim];
        for b in 0..m {
            for d in 0..x_dim {
                let c = (next() - 0.5) * 4.0;
                let w = next() * 2.0 + 0.05;
                xl[b * x_dim + d] = c - w;
                xu[b * x_dim + d] = c + w;
            }
        }
        // Entry 11: zero-width box (the verified-collapse shape).
        for d in 0..x_dim {
            xl[11 * x_dim + d] = 0.25;
            xu[11 * x_dim + d] = 0.25;
        }
        // Entry 12: infinite lower bound on dim 0 — drives the NaN candidate
        // reconstruction (dm - a*xhat + |a|*eps with inf snapshots) and the
        // revert-to-row-original lane.
        xl[12 * x_dim] = f32::NEG_INFINITY;
        xu[12 * x_dim] = 0.0;

        let mut a = vec![0f32; m * x_dim];
        for b in 0..m {
            for d in 0..x_dim {
                let r = next();
                a[b * x_dim + d] = if r < 0.15 {
                    0.0
                } else if r < 0.25 {
                    1e-11 * (next() - 0.5).signum() // near-zero sentinel stripe
                } else {
                    (next() - 0.5) * 3.0
                };
            }
        }
        // Entry 13: NaN coefficient on dim 2.
        a[13 * x_dim + 2] = f32::NAN;
        // Entry 12: mixed-sign coefficients against the infinite-bound box.
        a[12 * x_dim] = 1.0;
        a[12 * x_dim + 1] = -1.0;

        let mut bias = vec![0f32; m];
        let mut thr = vec![0f32; m];
        for b in 0..m {
            bias[b] = (next() - 0.5) * 2.0;
            thr[b] = (next() - 0.5) * 0.2;
        }
        // Aggressive rows so several entries latch infeasible; NaN / ±inf bias.
        bias[3] = 50.0;
        bias[9] = f32::NAN;
        bias[10] = f32::INFINITY;
        bias[8] = f32::NEG_INFINITY;

        for &is_lower in &[true, false] {
            for &num_iterations in &[1usize, 3, 20] {
                // Reference: the fast path with (m, 1, x_dim) tensors.
                let x_l_ref = ArrayD::from_shape_vec(vec![m, x_dim], xl.clone()).unwrap();
                let x_u_ref = ArrayD::from_shape_vec(vec![m, x_dim], xu.clone()).unwrap();
                let l_a_ref = ArrayD::from_shape_vec(vec![m, 1, x_dim], a.clone()).unwrap();
                let lbias_ref = ArrayD::from_shape_vec(vec![m, 1], bias.clone()).unwrap();
                let thr_ref = ArrayD::from_shape_vec(vec![m, 1], thr.clone()).unwrap();
                let reference = relaxed_clip_internal(
                    &x_l_ref,
                    &x_u_ref,
                    &l_a_ref,
                    &lbias_ref,
                    &thr_ref,
                    RelaxedClipOptions {
                        num_iterations,
                        is_lower,
                        preserve_infeasible: true,
                    },
                );

                let mut cl = xl.clone();
                let mut cu = xu.clone();
                let mut scratch = SingleSpecRowClipScratch::new();
                let core = relaxed_clip_single_spec_row_fast(
                    &mut cl,
                    &mut cu,
                    &a,
                    &bias,
                    &thr,
                    m,
                    x_dim,
                    num_iterations,
                    is_lower,
                    &mut scratch,
                );

                match reference {
                    Ok((rl, ru, rv)) => {
                        core.expect("core must succeed when the reference does");
                        assert_eq!(
                            rv, scratch.row_verified,
                            "row latch mismatch (is_lower={is_lower}, iters={num_iterations})"
                        );
                        let rl = rl.as_slice().unwrap();
                        let ru = ru.as_slice().unwrap();
                        for i in 0..m * x_dim {
                            assert_eq!(
                                rl[i].to_bits(),
                                cl[i].to_bits(),
                                "x_l bit mismatch at {i} (is_lower={is_lower}, iters={num_iterations}): ref={} core={}",
                                rl[i],
                                cl[i]
                            );
                            assert_eq!(
                                ru[i].to_bits(),
                                cu[i].to_bits(),
                                "x_u bit mismatch at {i} (is_lower={is_lower}, iters={num_iterations}): ref={} core={}",
                                ru[i],
                                cu[i]
                            );
                        }
                    }
                    Err(ref_err) => {
                        let core_err = core.expect_err("core must fail when the reference does");
                        assert_eq!(
                            ref_err.to_string(),
                            core_err.to_string(),
                            "error mismatch (is_lower={is_lower}, iters={num_iterations})"
                        );
                    }
                }
            }
        }
    }

    /// The single-spec row core must reject inverted input boxes with the
    /// reference's exact error (first failing entry, lowest dim).
    #[ntest::timeout(10000)]
    #[test]
    fn test_relaxed_clip_single_spec_row_inverted_input_error() {
        let m = 2usize;
        let x_dim = 2usize;
        let mut xl = vec![0.0, 0.0, 0.5, 0.0];
        let mut xu = vec![1.0, 1.0, 0.25, 1.0]; // entry 1 dim 0 inverted
        let a = vec![1.0; m * x_dim];
        let bias = vec![0.0; m];
        let thr = vec![0.0; m];
        let mut scratch = SingleSpecRowClipScratch::new();
        let err = relaxed_clip_single_spec_row_fast(
            &mut xl,
            &mut xu,
            &a,
            &bias,
            &thr,
            m,
            x_dim,
            1,
            true,
            &mut scratch,
        )
        .expect_err("inverted input must error");
        assert_eq!(
            err.to_string(),
            ny_core::NyError::InvalidSpec("relaxed_clip: x_l > x_u at batch=1 dim=0".to_string())
                .to_string()
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_concretize_bounds_simple() {
        // Single batch, single spec, 2D input
        let x_hat = array![[0.5, 0.5]].into_dyn();
        let eps = array![[0.5, 0.5]].into_dyn();
        let l_a = array![[[1.0, 1.0]]].into_dyn(); // x1 + x2
        let lbias = array![[0.0]].into_dyn();

        // Lower bound: l_a·xhat - |l_a|·eps + lbias
        // = (1*0.5 + 1*0.5) - (1*0.5 + 1*0.5) + 0 = 1.0 - 1.0 = 0.0
        let lb = concretize_bounds(&x_hat, &eps, &l_a, &lbias, true);
        assert!((lb[[0, 0]] - 0.0).abs() < 1e-6);

        // Upper bound: l_a·xhat + |l_a|·eps + lbias
        // = (1*0.5 + 1*0.5) + (1*0.5 + 1*0.5) + 0 = 1.0 + 1.0 = 2.0
        let ub = concretize_bounds(&x_hat, &eps, &l_a, &lbias, false);
        assert!((ub[[0, 0]] - 2.0).abs() < 1e-6);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_concretize_bounds_with_bias() {
        let x_hat = array![[0.0, 0.0]].into_dyn();
        let eps = array![[1.0, 1.0]].into_dyn();
        let l_a = array![[[1.0, -1.0]]].into_dyn(); // x1 - x2
        let lbias = array![[5.0]].into_dyn();

        // Lower bound: l_a·xhat - |l_a|·eps + lbias
        // = (1*0 - 1*0) - (1*1 + 1*1) + 5 = 0 - 2 + 5 = 3.0
        let lb = concretize_bounds(&x_hat, &eps, &l_a, &lbias, true);
        assert!((lb[[0, 0]] - 3.0).abs() < 1e-6);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relaxed_clip_no_change() {
        // When constraint is already satisfied everywhere, bounds don't change
        let x_l = array![[0.0, 0.0]].into_dyn();
        let x_u = array![[1.0, 1.0]].into_dyn();
        // Constraint: -x1 - x2 <= 0 (always satisfied for x >= 0)
        let l_a = array![[[-1.0, -1.0]]].into_dyn();
        let lbias = array![[0.0]].into_dyn();
        let thresholds = array![[0.0]].into_dyn();

        let (new_l, new_u) = relaxed_clip(&x_l, &x_u, &l_a, &lbias, &thresholds, 1, true).unwrap();

        // Bounds should remain approximately the same
        assert!((new_l[[0, 0]] - 0.0).abs() < 1e-5);
        assert!((new_l[[0, 1]] - 0.0).abs() < 1e-5);
        assert!((new_u[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((new_u[[0, 1]] - 1.0).abs() < 1e-5);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relaxed_clip_shape_mismatch() {
        let x_l = array![[0.0, 0.0]].into_dyn();
        // x_u has mismatched x_dim
        let x_u = array![[0.0, 0.0, 0.0]].into_dyn();
        let l_a = array![[[1.0, 1.0]]].into_dyn();
        let lbias = array![[0.0]].into_dyn();
        let thresholds = array![[0.0]].into_dyn();

        let result = relaxed_clip(&x_l, &x_u, &l_a, &lbias, &thresholds, 1, true);
        assert!(result.is_err(), "Expected shape mismatch error");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relaxed_clip_inverted_bounds_returns_invalid_spec() {
        let x_l = array![[1.0, 0.0]].into_dyn();
        let x_u = array![[0.0, 1.0]].into_dyn();
        let l_a = array![[[1.0, 1.0]]].into_dyn();
        let lbias = array![[0.0]].into_dyn();
        let thresholds = array![[0.0]].into_dyn();

        let err = relaxed_clip(&x_l, &x_u, &l_a, &lbias, &thresholds, 1, true)
            .expect_err("inverted input bounds must be rejected");

        assert!(
            matches!(err, ny_core::NyError::InvalidSpec(_)),
            "expected InvalidSpec for inverted bounds, got {err:?}"
        );
        assert!(
            err.to_string().contains("relaxed_clip: x_l > x_u"),
            "unexpected error message: {err}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relaxed_clip_tightens_bounds() {
        // Constraint that should tighten bounds
        let x_l = array![[0.0, 0.0]].into_dyn();
        let x_u = array![[10.0, 10.0]].into_dyn();
        // Constraint: x1 + x2 - 5 <= 0, meaning x1 + x2 <= 5
        // This should clip the upper bounds
        let l_a = array![[[1.0, 1.0]]].into_dyn();
        let lbias = array![[-5.0]].into_dyn();
        let thresholds = array![[0.0]].into_dyn();

        let (new_l, new_u) = relaxed_clip(&x_l, &x_u, &l_a, &lbias, &thresholds, 1, true).unwrap();

        // Lower bounds should stay at 0
        assert!((new_l[[0, 0]] - 0.0).abs() < 1e-5);
        assert!((new_l[[0, 1]] - 0.0).abs() < 1e-5);

        // Upper bounds should be tightened (constraint limits sum to 5)
        assert!(new_u[[0, 0]] < 10.0, "Upper bound should be tightened");
        assert!(new_u[[0, 1]] < 10.0, "Upper bound should be tightened");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relaxed_clip_preserves_valid_bounds() {
        // Even with aggressive constraints, bounds should stay valid (l <= u)
        let x_l = array![[0.0, 0.0]].into_dyn();
        let x_u = array![[1.0, 1.0]].into_dyn();
        // Very aggressive constraint
        let l_a = array![[[10.0, 10.0]]].into_dyn();
        let lbias = array![[-100.0]].into_dyn();
        let thresholds = array![[0.0]].into_dyn();

        let (new_l, new_u) = relaxed_clip(&x_l, &x_u, &l_a, &lbias, &thresholds, 1, true).unwrap();

        // Bounds should remain valid
        assert!(new_l[[0, 0]] <= new_u[[0, 0]], "Bounds must be valid");
        assert!(new_l[[0, 1]] <= new_u[[0, 1]], "Bounds must be valid");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relaxed_clip_multi_spec_infeasible_marks_verified() {
        let x_l = array![[0.0]].into_dyn();
        let x_u = array![[1.0]].into_dyn();
        // Contradictory pair:
        // 1. x <= 0.2
        // 2. x >= 0.8  ->  -x + 0.8 <= 0
        let l_a = array![[[1.0], [-1.0]]].into_dyn();
        let lbias = array![[-0.2, 0.8]].into_dyn();
        let thresholds = array![[0.0, 0.0]].into_dyn();

        let (new_l, new_u, verified_by_clip) =
            relaxed_clip_with_infeasible_mask(&x_l, &x_u, &l_a, &lbias, &thresholds, 1, true)
                .unwrap();

        assert_eq!(verified_by_clip, vec![true]);
        assert!(
            new_l[[0, 0]] > new_u[[0, 0]],
            "joint multi-spec clipping should surface infeasible bounds"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relaxed_clip_multi_iter_infeasible_sentinel_stays_allowed() {
        let x_l = array![[0.0]].into_dyn();
        let x_u = array![[1.0]].into_dyn();
        let l_a = array![[[1.0], [-1.0]]].into_dyn();
        let lbias = array![[-0.2, 0.8]].into_dyn();
        let thresholds = array![[0.0, 0.0]].into_dyn();

        let (new_l, new_u, verified_by_clip) =
            relaxed_clip_with_infeasible_mask(&x_l, &x_u, &l_a, &lbias, &thresholds, 2, true)
                .expect("verified sentinel batches must survive later iterations");

        assert_eq!(verified_by_clip, vec![true]);
        assert!(
            new_l[[0, 0]] > new_u[[0, 0]],
            "verified sentinel should remain inverted for preserved infeasible batches"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relaxed_clip_multiple_specs() {
        // Multiple specifications should all contribute
        let x_l = array![[0.0, 0.0]].into_dyn();
        let x_u = array![[10.0, 10.0]].into_dyn();
        // Two constraints:
        // 1. x1 <= 5 (represented as x1 - 5 <= 0)
        // 2. x2 <= 3 (represented as x2 - 3 <= 0)
        let l_a = array![[[1.0, 0.0], [0.0, 1.0]]].into_dyn();
        let lbias = array![[-5.0, -3.0]].into_dyn();
        let thresholds = array![[0.0, 0.0]].into_dyn();

        let (new_l, new_u) = relaxed_clip(&x_l, &x_u, &l_a, &lbias, &thresholds, 1, true).unwrap();

        // Lower bounds should remain at 0 for upper-only constraints
        assert!((new_l[[0, 0]] - 0.0).abs() < 1e-5);
        assert!((new_l[[0, 1]] - 0.0).abs() < 1e-5);

        // x1 upper should be clipped to ~5 (LP dual bound; 1e-2 for LP solver precision)
        assert!(
            (new_u[[0, 0]] - 5.0).abs() < 1e-2,
            "x1 upper should be ~5, got {}",
            new_u[[0, 0]]
        );
        // x2 upper should be clipped to ~3 (LP dual bound; 1e-2 for LP solver precision)
        assert!(
            (new_u[[0, 1]] - 3.0).abs() < 1e-2,
            "x2 upper should be ~3, got {}",
            new_u[[0, 1]]
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relaxed_clip_batch() {
        // Test with multiple batches
        let x_l = array![[0.0, 0.0], [0.0, 0.0]].into_dyn();
        let x_u = array![[10.0, 10.0], [20.0, 20.0]].into_dyn();
        // Same constraint for both batches
        let l_a = array![[[1.0, 1.0]], [[1.0, 1.0]]].into_dyn();
        let lbias = array![[-5.0], [-10.0]].into_dyn();
        let thresholds = array![[0.0], [0.0]].into_dyn();

        let (new_l, new_u) = relaxed_clip(&x_l, &x_u, &l_a, &lbias, &thresholds, 1, true).unwrap();

        // Lower bounds should remain at 0 for upper-only constraints
        assert!((new_l[[0, 0]] - 0.0).abs() < 1e-5);
        assert!((new_l[[0, 1]] - 0.0).abs() < 1e-5);
        assert!((new_l[[1, 0]] - 0.0).abs() < 1e-5);
        assert!((new_l[[1, 1]] - 0.0).abs() < 1e-5);

        // First batch should have tighter bounds
        assert!(new_u[[0, 0]] < 10.0);
        // Second batch should also be tightened
        assert!(new_u[[1, 0]] < 20.0);
    }

    // ========================================================================
    // Edge Case Tests (Part of #280)
    // ========================================================================

    #[ntest::timeout(10000)]
    #[test]
    fn test_relaxed_clip_multi_iteration() {
        // Multi-iteration clipping should produce tighter bounds than single iteration
        // because each iteration refines using updated bounds.
        let x_l = array![[0.0, 0.0]].into_dyn();
        let x_u = array![[10.0, 10.0]].into_dyn();
        // Constraint: x1 + x2 <= 5
        let l_a = array![[[1.0, 1.0]]].into_dyn();
        let lbias = array![[-5.0]].into_dyn();
        let thresholds = array![[0.0]].into_dyn();

        // Single iteration
        let (new_l_1, new_u_1) =
            relaxed_clip(&x_l, &x_u, &l_a, &lbias, &thresholds, 1, true).unwrap();

        // Multiple iterations
        let (new_l_3, new_u_3) =
            relaxed_clip(&x_l, &x_u, &l_a, &lbias, &thresholds, 3, true).unwrap();

        // Multi-iteration should give bounds at least as tight (usually tighter)
        assert!(
            new_u_3[[0, 0]] <= new_u_1[[0, 0]] + 1e-5,
            "Multi-iter upper bound should be <= single-iter"
        );
        assert!(
            new_u_3[[0, 1]] <= new_u_1[[0, 1]] + 1e-5,
            "Multi-iter upper bound should be <= single-iter"
        );
        assert!(
            new_l_3[[0, 0]] >= new_l_1[[0, 0]] - 1e-5,
            "Multi-iter lower bound should be >= single-iter"
        );
        assert!(
            new_l_3[[0, 1]] >= new_l_1[[0, 1]] - 1e-5,
            "Multi-iter lower bound should be >= single-iter"
        );

        // Bounds should still be valid
        assert!(
            new_l_3[[0, 0]] <= new_u_3[[0, 0]],
            "Bounds must remain valid"
        );
        assert!(
            new_l_3[[0, 1]] <= new_u_3[[0, 1]],
            "Bounds must remain valid"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relaxed_clip_negative_coefficients() {
        // Test with negative coefficients - these give candidates for lower bound tightening.
        // Constraint: -x1 <= -2 (i.e., x1 >= 2) should raise x1 lower bound.
        let x_l = array![[0.0, 0.0]].into_dyn();
        let x_u = array![[10.0, 10.0]].into_dyn();
        // Single variable constraint: -x1 - 2 <= 0, meaning x1 >= 2
        let l_a = array![[[-1.0, 0.0]]].into_dyn();
        let lbias = array![[-2.0]].into_dyn();
        let thresholds = array![[0.0]].into_dyn();

        let (new_l, new_u) = relaxed_clip(&x_l, &x_u, &l_a, &lbias, &thresholds, 1, true).unwrap();

        // x1 lower bound should stay at or above 0 (original bound)
        // Note: relaxed clipping may not tighten in all cases depending on the
        // concrete bound computation. The key property is soundness (no expansion).
        assert!(
            new_l[[0, 0]] >= -0.01,
            "x1 lower should not decrease below 0, got {}",
            new_l[[0, 0]]
        );

        // x2 should be unchanged (zero coefficient)
        assert!(
            (new_l[[0, 1]] - 0.0).abs() < 0.01,
            "x2 lower should stay at 0"
        );

        // Upper bounds should remain unchanged (negative coefficient doesn't affect upper)
        assert!(
            (new_u[[0, 0]] - 10.0).abs() < 0.01,
            "x1 upper should stay at 10"
        );

        // Bounds must remain valid
        assert!(new_l[[0, 0]] <= new_u[[0, 0]], "Bounds must be valid");
        assert!(new_l[[0, 1]] <= new_u[[0, 1]], "Bounds must be valid");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relaxed_clip_mixed_sign_constraints() {
        // Test that bounds remain valid with mixed positive/negative coefficients.
        // The relaxed clipping algorithm computes tight bounds using centroid-based
        // relaxation, which may not tighten in all configurations.
        let x_l = array![[0.0, 0.0]].into_dyn();
        let x_u = array![[10.0, 10.0]].into_dyn();
        // Two constraints with different signs:
        // 1. x1 - x2 <= 3 (positive on x1, negative on x2)
        // 2. -x1 + x2 <= 3 (negative on x1, positive on x2)
        let l_a = array![[[1.0, -1.0], [-1.0, 1.0]]].into_dyn();
        let lbias = array![[-3.0, -3.0]].into_dyn();
        let thresholds = array![[0.0, 0.0]].into_dyn();

        let (new_l, new_u) = relaxed_clip(&x_l, &x_u, &l_a, &lbias, &thresholds, 1, true).unwrap();

        // Primary check: bounds must remain valid
        assert!(new_l[[0, 0]] <= new_u[[0, 0]], "Bounds must be valid");
        assert!(new_l[[0, 1]] <= new_u[[0, 1]], "Bounds must be valid");

        // Bounds should not expand beyond original
        assert!(
            new_l[[0, 0]] >= 0.0 - 1e-5,
            "Lower bound should not decrease"
        );
        assert!(
            new_l[[0, 1]] >= 0.0 - 1e-5,
            "Lower bound should not decrease"
        );
        assert!(
            new_u[[0, 0]] <= 10.0 + 1e-5,
            "Upper bound should not increase"
        );
        assert!(
            new_u[[0, 1]] <= 10.0 + 1e-5,
            "Upper bound should not increase"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relaxed_clip_zero_coefficient_dimension() {
        // Test handling of dimensions with zero coefficient (no constraint on that dim)
        let x_l = array![[0.0, 0.0, 0.0]].into_dyn();
        let x_u = array![[10.0, 10.0, 10.0]].into_dyn();
        // Constraint: x1 + 0*x2 + x3 <= 5 (x2 has zero coefficient)
        let l_a = array![[[1.0, 0.0, 1.0]]].into_dyn();
        let lbias = array![[-5.0]].into_dyn();
        let thresholds = array![[0.0]].into_dyn();

        let (new_l, new_u) = relaxed_clip(&x_l, &x_u, &l_a, &lbias, &thresholds, 1, true).unwrap();

        // x2 should not be affected (zero coefficient means no constraint on that dimension)
        assert!(
            (new_l[[0, 1]] - 0.0).abs() < 0.01,
            "x2 lower should stay at 0"
        );
        assert!(
            (new_u[[0, 1]] - 10.0).abs() < 0.01,
            "x2 upper should stay at 10, got {}",
            new_u[[0, 1]]
        );

        // x1 and x3 should be tightened
        assert!(new_u[[0, 0]] < 10.0, "x1 upper should be tightened");
        assert!(new_u[[0, 2]] < 10.0, "x3 upper should be tightened");
    }

    /// Regression test for #2963: NaN in CROWN coefficients must produce
    /// conservative -inf/+inf bounds instead of propagating NaN.
    #[ntest::timeout(10000)]
    #[test]
    fn test_concretize_bounds_nan_coefficient_produces_conservative_bounds() {
        // NaN in l_a coefficient — concretize_bounds must not output NaN.
        let x_hat = array![[0.5, 0.5]].into_dyn();
        let eps = array![[0.5, 0.5]].into_dyn();
        // First spec has NaN coefficient, second spec is clean
        let l_a = array![[[f32::NAN, 1.0], [1.0, 1.0]]].into_dyn();
        let lbias = array![[0.0, 0.0]].into_dyn();

        // Lower bound: NaN spec should produce -inf, clean spec should be finite
        let lb = concretize_bounds(&x_hat, &eps, &l_a, &lbias, true);
        assert!(
            lb[[0, 0]] == f32::NEG_INFINITY,
            "NaN coefficient should produce -inf for lower bound, got {}",
            lb[[0, 0]]
        );
        assert!(
            lb[[0, 1]].is_finite(),
            "Clean spec should produce finite lower bound, got {}",
            lb[[0, 1]]
        );

        // Upper bound: NaN spec should produce +inf, clean spec should be finite
        let ub = concretize_bounds(&x_hat, &eps, &l_a, &lbias, false);
        assert!(
            ub[[0, 0]] == f32::INFINITY,
            "NaN coefficient should produce +inf for upper bound, got {}",
            ub[[0, 0]]
        );
        assert!(
            ub[[0, 1]].is_finite(),
            "Clean spec should produce finite upper bound, got {}",
            ub[[0, 1]]
        );
    }

    /// Regression test for #2963: NaN in bias must produce conservative bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn test_concretize_bounds_nan_bias_produces_conservative_bounds() {
        let x_hat = array![[0.5, 0.5]].into_dyn();
        let eps = array![[0.5, 0.5]].into_dyn();
        let l_a = array![[[1.0, 1.0]]].into_dyn();
        let lbias = array![[f32::NAN]].into_dyn();

        let lb = concretize_bounds(&x_hat, &eps, &l_a, &lbias, true);
        assert!(
            lb[[0, 0]] == f32::NEG_INFINITY,
            "NaN bias should produce -inf for lower bound, got {}",
            lb[[0, 0]]
        );

        let ub = concretize_bounds(&x_hat, &eps, &l_a, &lbias, false);
        assert!(
            ub[[0, 0]] == f32::INFINITY,
            "NaN bias should produce +inf for upper bound, got {}",
            ub[[0, 0]]
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relaxed_clip_near_zero_coefficient() {
        // Test handling of very small (near-zero) coefficients
        let x_l = array![[0.0, 0.0]].into_dyn();
        let x_u = array![[10.0, 10.0]].into_dyn();
        // Constraint with one very small coefficient: 1e-12 * x1 + x2 <= 5
        let l_a = array![[[1e-12, 1.0]]].into_dyn();
        let lbias = array![[-5.0]].into_dyn();
        let thresholds = array![[0.0]].into_dyn();

        let (new_l, new_u) = relaxed_clip(&x_l, &x_u, &l_a, &lbias, &thresholds, 1, true).unwrap();

        // x1 should not be significantly affected (near-zero coefficient)
        assert!(
            new_u[[0, 0]] > 9.0,
            "x1 upper should stay near 10 with tiny coefficient, got {}",
            new_u[[0, 0]]
        );

        // x2 should be tightened to ~5
        assert!(
            new_u[[0, 1]] < 6.0,
            "x2 upper should be clipped to ~5, got {}",
            new_u[[0, 1]]
        );

        // Bounds should remain valid
        assert!(new_l[[0, 0]] <= new_u[[0, 0]], "Bounds must be valid");
        assert!(new_l[[0, 1]] <= new_u[[0, 1]], "Bounds must be valid");
    }
}
