// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched relaxed clipping for input-split BaB survivors.
//!
//! Applies relaxed clip across N children simultaneously, iterating over
//! threshold rows sequentially (preserving the per-threshold tightening
//! semantics of `clip_with_precomputed_linear`).
//!
//! Part of #4366 Packet B.

use ndarray::{Array2, Array3, ArrayD, IxDyn};
use ny_core::{NyError, Result};

use crate::bounds::LinearBounds;
use crate::relaxed_clip::{
    relaxed_clip_single_spec_row_fast, relaxed_clip_with_infeasible_mask, SingleSpecRowClipScratch,
};

/// Result of batched relaxed clipping across N children.
pub(super) struct BatchedClipResult {
    /// Clipped flat lower arrays, one per child.
    pub(super) clipped_lowers: Vec<ArrayD<f32>>,
    /// Clipped flat upper arrays, one per child.
    pub(super) clipped_uppers: Vec<ArrayD<f32>>,
    /// Per-child verified flags (true if clipping proved the domain infeasible).
    pub(super) verified: Vec<bool>,
}

/// Batch relaxed clip across N children using their precomputed LinearBounds.
///
/// Applies each threshold row sequentially (matching the per-threshold loop in
/// `clip_with_precomputed_linear`), but batches the N children into a single
/// `relaxed_clip` call per threshold. This avoids N individual flatten+reshape
/// cycles per threshold.
///
/// # Arguments
///
/// * `flat_lowers` - Flat 1D lower arrays per child, all same length
/// * `flat_uppers` - Flat 1D upper arrays per child, all same length
/// * `linear_bounds_list` - LinearBounds per child (from parent CROWN pass)
/// * `thresholds` - Threshold values per objective row
/// * `verify_upper_bound` - If true, clip using upper bound constraints
/// * `relaxed_clip_iterations` - Number of iterative refinement passes
///
/// Reference: `clip_with_precomputed_linear` in `relaxed_clip.rs:160-250`
#[allow(clippy::too_many_arguments)]
pub(super) fn batched_relaxed_clip_from_flat(
    flat_lowers: &[ArrayD<f32>],
    flat_uppers: &[ArrayD<f32>],
    linear_bounds_list: &[&LinearBounds],
    thresholds: &[f32],
    clause_sizes: &[usize],
    verify_upper_bound: bool,
    relaxed_clip_iterations: usize,
) -> Result<BatchedClipResult> {
    let n = flat_lowers.len();
    if n == 0 {
        return Ok(BatchedClipResult {
            clipped_lowers: Vec::new(),
            clipped_uppers: Vec::new(),
            verified: Vec::new(),
        });
    }
    if flat_uppers.len() != n || linear_bounds_list.len() != n {
        return Err(NyError::InvalidSpec(format!(
            "batched_relaxed_clip_from_flat: count mismatch: lowers={} uppers={} linear_bounds={}",
            n,
            flat_uppers.len(),
            linear_bounds_list.len()
        )));
    }

    let x_dim = flat_lowers[0].len();

    // Stack flat arrays into (N, x_dim) matrices.
    let mut lower_data = Vec::with_capacity(n * x_dim);
    let mut upper_data = Vec::with_capacity(n * x_dim);
    for i in 0..n {
        if flat_lowers[i].len() != x_dim || flat_uppers[i].len() != x_dim {
            return Err(NyError::InvalidSpec(format!(
                "batched_relaxed_clip_from_flat: child {} dim mismatch: lower={} upper={} expected={}",
                i,
                flat_lowers[i].len(),
                flat_uppers[i].len(),
                x_dim
            )));
        }
        lower_data.extend(flat_lowers[i].iter());
        upper_data.extend(flat_uppers[i].iter());
    }

    let orig_l = ArrayD::from_shape_vec(IxDyn(&[n, x_dim]), lower_data)
        .map_err(|e| NyError::InvalidSpec(format!("batched_clip: reshape x_l: {}", e)))?;
    let orig_u = ArrayD::from_shape_vec(IxDyn(&[n, x_dim]), upper_data)
        .map_err(|e| NyError::InvalidSpec(format!("batched_clip: reshape x_u: {}", e)))?;

    // #disj-cross-clause-clip-unsat: the CLAUSE-AWARE grouped driver runs the
    // sequential-threshold core once per clause from these originals and unions
    // the per-clause clipped boxes; single-clause specs delegate to the
    // whole-spec core bit-for-bit.
    let (x_l, x_u, verified_by_clip) = batched_relaxed_clip_core_grouped(
        &orig_l,
        &orig_u,
        linear_bounds_list,
        thresholds,
        clause_sizes,
        verify_upper_bound,
        relaxed_clip_iterations,
        n,
        x_dim,
    )?;

    // Split batched results back into per-child arrays.
    let mut clipped_lowers = Vec::with_capacity(n);
    let mut clipped_uppers = Vec::with_capacity(n);
    for i in 0..n {
        let row_l: Vec<f32> = (0..x_dim).map(|d| x_l[[i, d]]).collect();
        let row_u: Vec<f32> = (0..x_dim).map(|d| x_u[[i, d]]).collect();

        clipped_lowers.push(ArrayD::from_shape_vec(IxDyn(&[x_dim]), row_l).map_err(|e| {
            NyError::InvalidSpec(format!("batched_clip: split lower[{}]: {}", i, e))
        })?);
        clipped_uppers.push(ArrayD::from_shape_vec(IxDyn(&[x_dim]), row_u).map_err(|e| {
            NyError::InvalidSpec(format!("batched_clip: split upper[{}]: {}", i, e))
        })?);
    }

    Ok(BatchedClipResult {
        clipped_lowers,
        clipped_uppers,
        verified: verified_by_clip,
    })
}

/// #lsnc-child-batch (S1): stacked-row entry for the batched relaxed clip.
///
/// Like [`batched_relaxed_clip_from_flat`] but consumes the PRE-STACKED
/// `(N, x_dim)` original child boxes directly (rows written contiguously by
/// the `ChildBatch` split kernel) and returns the stacked clipped arrays
/// without the per-child split-back. Runs the SAME clause-aware grouped driver
/// (`batched_relaxed_clip_core_grouped`) with `linear_bounds_list[i]` gathered
/// by the caller from the shared parent planes (`parent_idx`), so given
/// bit-identical rows and coefficient sources the result is bit-identical to
/// the flat entry. Parity: `test_child_batch_reorder_prescreen_parity_lsnc_s1`.
pub(super) fn batched_relaxed_clip_from_stacked(
    orig_l: &ArrayD<f32>,
    orig_u: &ArrayD<f32>,
    linear_bounds_list: &[&LinearBounds],
    thresholds: &[f32],
    clause_sizes: &[usize],
    verify_upper_bound: bool,
    relaxed_clip_iterations: usize,
) -> Result<(ArrayD<f32>, ArrayD<f32>, Vec<bool>)> {
    let shape = orig_l.shape();
    if shape.len() != 2 || orig_u.shape() != shape {
        return Err(NyError::InvalidSpec(format!(
            "batched_relaxed_clip_from_stacked: expected matching (N, x_dim) arrays, got {:?} and {:?}",
            shape,
            orig_u.shape()
        )));
    }
    let (n, x_dim) = (shape[0], shape[1]);
    if linear_bounds_list.len() != n {
        return Err(NyError::InvalidSpec(format!(
            "batched_relaxed_clip_from_stacked: children={} linear_bounds={}",
            n,
            linear_bounds_list.len()
        )));
    }
    if n == 0 {
        return Ok((orig_l.clone(), orig_u.clone(), Vec::new()));
    }

    batched_relaxed_clip_core_grouped(
        orig_l,
        orig_u,
        linear_bounds_list,
        thresholds,
        clause_sizes,
        verify_upper_bound,
        relaxed_clip_iterations,
        n,
        x_dim,
    )
}

/// Sequential-threshold relaxed-clip core shared by the flat and stacked
/// entries. `x_l`/`x_u` are the working `(N, x_dim)` boxes (initially equal to
/// `orig_l`/`orig_u`); `orig_l`/`orig_u` are the PRE-CLIP originals consumed by
/// the verified-child midpoint restore (#4367). Only the threshold rows in
/// `rows` are applied — SEQUENTIALLY (row k+1 sees row k's clipped box, I-A9) —
/// and `verified` latches monotonically. Restricting `rows` to a single
/// clause's span (`offset..offset+size`) is how the clause-aware grouped driver
/// intersects ONLY that clause's half-spaces (see
/// [`batched_relaxed_clip_core_grouped`]); passing the whole `0..thresholds.len()`
/// reproduces the historical whole-spec intersection bit-for-bit.
#[allow(clippy::too_many_arguments)]
fn batched_relaxed_clip_core(
    mut x_l: ArrayD<f32>,
    mut x_u: ArrayD<f32>,
    orig_l: &ArrayD<f32>,
    orig_u: &ArrayD<f32>,
    linear_bounds_list: &[&LinearBounds],
    thresholds: &[f32],
    verify_upper_bound: bool,
    relaxed_clip_iterations: usize,
    n: usize,
    x_dim: usize,
    rows: std::ops::Range<usize>,
) -> Result<(ArrayD<f32>, ArrayD<f32>, Vec<bool>)> {
    let is_lower = true;
    let mut verified_by_clip = vec![false; n];

    // Apply each threshold in `rows` sequentially (preserving per-threshold
    // tightening within the clause span).
    for row_idx in rows {
        let threshold = thresholds[row_idx];
        // Restore valid bounds for already-verified children before passing to
        // the next threshold's clip. When a prior threshold made a child
        // infeasible, `preserve_infeasible=true` left x_l > x_u. The next
        // `relaxed_clip_internal` creates a fresh `verified_by_clip` and would
        // error on the inverted bounds. Setting x_l = x_u for verified children
        // makes them zero-width (no further tightening possible) and avoids the
        // inversion check. Part of #4367.
        for i in 0..n {
            if verified_by_clip[i] {
                for d in 0..x_dim {
                    let lo = orig_l[[i, d]];
                    let hi = orig_u[[i, d]];
                    let mid = if lo.is_finite() && hi.is_finite() {
                        f32::midpoint(lo, hi)
                    } else if lo.is_finite() {
                        lo
                    } else if hi.is_finite() {
                        hi
                    } else {
                        0.0
                    };
                    x_l[[i, d]] = mid;
                    x_u[[i, d]] = mid;
                }
            }
        }

        // Build (N, 1, x_dim) coefficient tensor and (N, 1) bias tensor.
        let (l_a, lbias, thresh_mat) = build_batched_coefficients(
            linear_bounds_list,
            row_idx,
            threshold,
            verify_upper_bound,
            n,
            x_dim,
        )?;

        let (new_l, new_u, infeasible_mask) = relaxed_clip_with_infeasible_mask(
            &x_l,
            &x_u,
            &l_a.into_dyn(),
            &lbias.into_dyn(),
            &thresh_mat.into_dyn(),
            relaxed_clip_iterations,
            is_lower,
        )?;

        // Merge infeasible flags: once verified, stays verified.
        for (i, &inf) in infeasible_mask.iter().enumerate() {
            if inf {
                verified_by_clip[i] = true;
            }
        }

        x_l = new_l;
        x_u = new_u;
    }

    Ok((x_l, x_u, verified_by_clip))
}

/// #disj-cross-clause-clip-unsat: CLAUSE-AWARE driver over the sequential
/// relaxed-clip core (stacked coefficient path shared by the flat and stacked
/// entries).
///
/// The relaxed clip carves each threshold row's still-possibly-violating
/// half-space (the CROWN lower bound `L_r(x) <= t_r` region, a SUPERSET of the
/// true `f_r(x) <= t_r` violating region). For a single clause (a conjunction:
/// a counterexample must violate EVERY row) intersecting all its rows is exactly
/// right, and an empty intersection refutes the clause. But a multi-clause OR
/// counterexample only has to satisfy ONE clause, so intersecting rows ACROSS
/// clauses (the historical whole-spec pass) keeps only the sub-box where every
/// clause is simultaneously possibly-satisfiable and DISCARDS the sub-boxes that
/// satisfy a single clause — genuine counterexamples (the lsnc false-unsat).
///
/// This driver instead clips EACH clause independently from the ORIGINAL box
/// (intersecting only that clause's rows), then:
///   * marks a child `verified` (UNSAT) iff EVERY clause's within-clause
///     intersection is empty — no clause can be satisfied anywhere in the box;
///   * carries forward the per-child UNION bounding box of the clauses that
///     remain feasible, which ENCLOSES the union of the per-clause retained
///     regions and hence every counterexample, so no counterexample is ever
///     discarded and the downstream `concretize_postclip_lower_bounds` ->
///     `disjunctive_domain_verified` re-bound stays a true over-approximation.
///
/// Single-clause (`clause_sizes.len() <= 1`) delegates to the whole-spec core
/// verbatim, so the conjunctive lanes (acasxu conjunctive, nn4sys, ...) are
/// bit-identical to the pre-clause-aware behavior.
#[allow(clippy::too_many_arguments)]
fn batched_relaxed_clip_core_grouped(
    orig_l: &ArrayD<f32>,
    orig_u: &ArrayD<f32>,
    linear_bounds_list: &[&LinearBounds],
    thresholds: &[f32],
    clause_sizes: &[usize],
    verify_upper_bound: bool,
    relaxed_clip_iterations: usize,
    n: usize,
    x_dim: usize,
) -> Result<(ArrayD<f32>, ArrayD<f32>, Vec<bool>)> {
    // Single clause (or degenerate): the whole-spec sequential pass, bit-identical
    // to the pre-clause-aware core.
    if clause_sizes.len() <= 1 {
        return batched_relaxed_clip_core(
            orig_l.clone(),
            orig_u.clone(),
            orig_l,
            orig_u,
            linear_bounds_list,
            thresholds,
            verify_upper_bound,
            relaxed_clip_iterations,
            n,
            x_dim,
            0..thresholds.len(),
        );
    }

    // Multi-clause OR: clip each clause independently from the original box.
    let mut union_l = ArrayD::<f32>::from_elem(IxDyn(&[n, x_dim]), f32::INFINITY);
    let mut union_u = ArrayD::<f32>::from_elem(IxDyn(&[n, x_dim]), f32::NEG_INFINITY);
    // A child stays `all_infeasible` only while every clause seen so far is empty
    // for it; `any_kept` records whether at least one clause contributed a region.
    let mut all_infeasible = vec![true; n];
    let mut any_kept = vec![false; n];

    let mut offset = 0usize;
    for &size in clause_sizes {
        let end = offset + size;
        let (cl, cu, cv) = batched_relaxed_clip_core(
            orig_l.clone(),
            orig_u.clone(),
            orig_l,
            orig_u,
            linear_bounds_list,
            thresholds,
            verify_upper_bound,
            relaxed_clip_iterations,
            n,
            x_dim,
            offset..end,
        )?;
        for k in 0..n {
            if cv[k] {
                // This clause is refuted for child k (its within-clause
                // intersection is empty): it contributes no counterexample
                // region, so it is excluded from the union.
                continue;
            }
            all_infeasible[k] = false;
            any_kept[k] = true;
            // A kept (non-verified) clause always leaves a valid finite box
            // (the core reverts to the original on NaN and only inverts bounds
            // when it latches `verified`), so the min/max is over finite values.
            for d in 0..x_dim {
                let l = cl[[k, d]];
                let u = cu[[k, d]];
                if l < union_l[[k, d]] {
                    union_l[[k, d]] = l;
                }
                if u > union_u[[k, d]] {
                    union_u[[k, d]] = u;
                }
            }
        }
        offset = end;
    }

    // Children with every clause refuted are reported verified, so the union box
    // is unused; restore the original box so no downstream reader ever observes
    // the +inf/-inf sentinel.
    for k in 0..n {
        if !any_kept[k] {
            for d in 0..x_dim {
                union_l[[k, d]] = orig_l[[k, d]];
                union_u[[k, d]] = orig_u[[k, d]];
            }
        }
    }

    Ok((union_l, union_u, all_infeasible))
}

/// #lsnc-clip-planes (S5): one parent's used-side clip plane — the coefficient
/// rows in the CLIP sign convention (`lower_a` rows verbatim for the
/// lower-bound direction; `upper_a` rows NEGATED for `verify_upper_bound`,
/// matching `build_batched_coefficients`), flat row-major `[nrows * x_dim]`.
/// Both children of one parent reference the SAME plane (the reference path
/// re-gathered the identical row per child per threshold).
pub(super) struct ParentClipPlane<'a> {
    pub(super) coeffs: std::borrow::Cow<'a, [f32]>,
    pub(super) nrows: usize,
}

/// #lsnc-clip-planes (S5): planes-based entry for the batched sequential
/// relaxed clip.
///
/// Semantically identical to [`batched_relaxed_clip_from_stacked`] fed
/// per-child `LinearBounds` whose used-side biases equal `bias_used` and whose
/// used-side coefficient rows equal the parent plane rows: same sequential
/// threshold order (row k+1 sees row k's clipped box, I-A9), same verified
/// midpoint collapse (#4367), same monotone latch. The per-threshold work is
/// restructured — coefficients are GATHERED from the shared parent planes via
/// `child_plane` into reused scratch instead of rebuilding an `(N, 1, x_dim)`
/// `Array3` per row, and the per-row clip runs through
/// [`relaxed_clip_single_spec_row_fast`] (the bit-parity `n_spec = 1` core
/// with caller-owned scratch) in place of a fresh
/// `relaxed_clip_with_infeasible_mask` allocation cycle. BIT-PARITY CLASS:
/// `test_batched_clip_planes_matches_stacked_s5`.
///
/// * `orig_l`/`orig_u`: `[m * x_dim]` pre-clip child boxes (row-major).
/// * `planes[child_plane[k]]`: child `k`'s parent plane.
/// * `bias_used`: `[m * n_thr]` per-child used-side biases in the clip sign
///   convention (upper-bound direction already negated), with any certified
///   coefficient error already folded over the child's own box (I-A10).
/// * rows `>= planes[..].nrows` follow the reference's out-of-range contract:
///   zero coefficients, zero bias, zero threshold (no constraint).
#[allow(clippy::too_many_arguments)]
pub(super) fn batched_relaxed_clip_from_planes(
    orig_l: &[f32],
    orig_u: &[f32],
    planes: &[ParentClipPlane<'_>],
    child_plane: &[usize],
    bias_used: &[f32],
    thresholds: &[f32],
    clause_sizes: &[usize],
    verify_upper_bound: bool,
    relaxed_clip_iterations: usize,
    m: usize,
    x_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<bool>)> {
    let n_thr = thresholds.len();
    if orig_l.len() != m * x_dim
        || orig_u.len() != m * x_dim
        || child_plane.len() != m
        || bias_used.len() != m * n_thr
    {
        return Err(NyError::InvalidSpec(format!(
            "batched_relaxed_clip_from_planes: shape mismatch (m={}, x_dim={}, n_thr={}, orig_l={}, orig_u={}, child_plane={}, bias_used={})",
            m,
            x_dim,
            n_thr,
            orig_l.len(),
            orig_u.len(),
            child_plane.len(),
            bias_used.len()
        )));
    }
    for (k, &slot) in child_plane.iter().enumerate() {
        let pl = planes.get(slot).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "batched_relaxed_clip_from_planes: child {} plane slot {} out of range",
                k, slot
            ))
        })?;
        if pl.coeffs.len() != pl.nrows * x_dim {
            return Err(NyError::InvalidSpec(format!(
                "batched_relaxed_clip_from_planes: plane {} coeffs len {} != nrows {} * x_dim {}",
                slot,
                pl.coeffs.len(),
                pl.nrows,
                x_dim
            )));
        }
    }

    // Reused per-row gather scratch (the historical path allocated a fresh
    // `(N, 1, x_dim)` Array3 + two `(N, 1)` Array2s per threshold row).
    let mut a_scr = vec![0f32; m * x_dim];
    let mut bias_scr = vec![0f32; m];
    let mut thr_scr = vec![0f32; m];
    let mut scratch = SingleSpecRowClipScratch::new();

    // Single clause (or degenerate): the whole-spec sequential pass, bit-identical
    // to the pre-clause-aware planes loop.
    if clause_sizes.len() <= 1 {
        let mut xl = orig_l.to_vec();
        let mut xu = orig_u.to_vec();
        let mut verified_by_clip = vec![false; m];
        clip_planes_rows(
            &mut xl,
            &mut xu,
            &mut verified_by_clip,
            orig_l,
            orig_u,
            planes,
            child_plane,
            bias_used,
            thresholds,
            0..n_thr,
            verify_upper_bound,
            relaxed_clip_iterations,
            m,
            x_dim,
            &mut a_scr,
            &mut bias_scr,
            &mut thr_scr,
            &mut scratch,
        )?;
        return Ok((xl, xu, verified_by_clip));
    }

    // Multi-clause OR (#disj-cross-clause-clip-unsat): clip each clause
    // independently from the original box, then union the feasible clauses'
    // boxes; verified iff every clause's within-clause intersection is empty.
    // See `batched_relaxed_clip_core_grouped` for the soundness argument.
    let mut union_l = vec![f32::INFINITY; m * x_dim];
    let mut union_u = vec![f32::NEG_INFINITY; m * x_dim];
    let mut all_infeasible = vec![true; m];
    let mut any_kept = vec![false; m];

    let mut offset = 0usize;
    for &size in clause_sizes {
        let end = offset + size;
        let mut xl = orig_l.to_vec();
        let mut xu = orig_u.to_vec();
        let mut verified_by_clip = vec![false; m];
        clip_planes_rows(
            &mut xl,
            &mut xu,
            &mut verified_by_clip,
            orig_l,
            orig_u,
            planes,
            child_plane,
            bias_used,
            thresholds,
            offset..end,
            verify_upper_bound,
            relaxed_clip_iterations,
            m,
            x_dim,
            &mut a_scr,
            &mut bias_scr,
            &mut thr_scr,
            &mut scratch,
        )?;
        for k in 0..m {
            if verified_by_clip[k] {
                continue;
            }
            all_infeasible[k] = false;
            any_kept[k] = true;
            let base = k * x_dim;
            for d in 0..x_dim {
                let l = xl[base + d];
                let u = xu[base + d];
                if l < union_l[base + d] {
                    union_l[base + d] = l;
                }
                if u > union_u[base + d] {
                    union_u[base + d] = u;
                }
            }
        }
        offset = end;
    }

    // All-clauses-refuted children keep the sentinel box (unused since verified);
    // restore the original box so no downstream reader observes +inf/-inf.
    for k in 0..m {
        if !any_kept[k] {
            let range = k * x_dim..(k + 1) * x_dim;
            union_l[range.clone()].copy_from_slice(&orig_l[range.clone()]);
            union_u[range.clone()].copy_from_slice(&orig_u[range]);
        }
    }

    Ok((union_l, union_u, all_infeasible))
}

/// #disj-cross-clause-clip-unsat: run the planes-based sequential relaxed clip
/// for the threshold rows in `rows` (a single clause's span, or the whole spec)
/// over the working box `xl`/`xu`, latching `verified_by_clip` monotonically.
/// Extracted verbatim from the historical whole-spec loop so the single-clause
/// path stays bit-identical; the clause-aware driver calls it once per clause
/// from a fresh copy of the original box.
#[allow(clippy::too_many_arguments)]
fn clip_planes_rows(
    xl: &mut [f32],
    xu: &mut [f32],
    verified_by_clip: &mut [bool],
    orig_l: &[f32],
    orig_u: &[f32],
    planes: &[ParentClipPlane<'_>],
    child_plane: &[usize],
    bias_used: &[f32],
    thresholds: &[f32],
    rows: std::ops::Range<usize>,
    verify_upper_bound: bool,
    relaxed_clip_iterations: usize,
    m: usize,
    x_dim: usize,
    a_scr: &mut [f32],
    bias_scr: &mut [f32],
    thr_scr: &mut [f32],
    scratch: &mut SingleSpecRowClipScratch,
) -> Result<()> {
    let n_thr = thresholds.len();
    for row_idx in rows {
        let threshold = thresholds[row_idx];
        // Verified-child midpoint collapse before the next row (#4367),
        // identical to `batched_relaxed_clip_core`.
        for i in 0..m {
            if verified_by_clip[i] {
                let base = i * x_dim;
                for d in 0..x_dim {
                    let lo = orig_l[base + d];
                    let hi = orig_u[base + d];
                    let mid = if lo.is_finite() && hi.is_finite() {
                        f32::midpoint(lo, hi)
                    } else if lo.is_finite() {
                        lo
                    } else if hi.is_finite() {
                        hi
                    } else {
                        0.0
                    };
                    xl[base + d] = mid;
                    xu[base + d] = mid;
                }
            }
        }

        // Gather this row's coefficients / bias / threshold per child from the
        // shared parent planes (`build_batched_coefficients` semantics,
        // including the out-of-range zero contract and threshold negation).
        let row_thr = if verify_upper_bound {
            -threshold
        } else {
            threshold
        };
        for k in 0..m {
            let pl = &planes[child_plane[k]];
            let dst = &mut a_scr[k * x_dim..(k + 1) * x_dim];
            if row_idx >= pl.nrows {
                dst.fill(0.0);
                bias_scr[k] = 0.0;
                thr_scr[k] = 0.0;
            } else {
                dst.copy_from_slice(&pl.coeffs[row_idx * x_dim..(row_idx + 1) * x_dim]);
                bias_scr[k] = bias_used[k * n_thr + row_idx];
                thr_scr[k] = row_thr;
            }
        }

        relaxed_clip_single_spec_row_fast(
            xl,
            xu,
            a_scr,
            bias_scr,
            thr_scr,
            m,
            x_dim,
            relaxed_clip_iterations,
            true, // is_lower, as in `batched_relaxed_clip_core`
            scratch,
        )?;

        // Merge infeasible flags: once verified, stays verified.
        for (i, &inf) in scratch.row_verified.iter().enumerate() {
            if inf {
                verified_by_clip[i] = true;
            }
        }
    }

    Ok(())
}

/// #lsnc-clip-planes (S5): planes-based post-clip concretize for ONE child —
/// the same per-row f64 accumulation, directed `next_down_f32` round, and
/// NaN→`-inf` degrade as `concretize_postclip_lower_bounds`
/// (`push_survivors.rs`), reading the parent plane rows (already in the clip
/// sign convention) + the child's folded used-side biases instead of a
/// per-child `LinearBounds`, and writing into a caller-reused buffer instead
/// of allocating per child. Coefficient error is already discharged into
/// `bias_row` (I-A10), matching the reference's fold-before-concretize
/// contract. Parity: `test_batched_clip_planes_matches_stacked_s5`.
///
/// [`concretize_postclip_lower_bounds`]:
/// super::disjunctive_multi_clause::push_survivors::concretize_postclip_lower_bounds
pub(super) fn concretize_postclip_lower_bounds_planes(
    clip_l_row: &[f32],
    clip_u_row: &[f32],
    plane: &ParentClipPlane<'_>,
    bias_row: &[f32],
    n_thresholds: usize,
    out: &mut Vec<(f32, f32)>,
) {
    let x_dim = clip_l_row.len();
    let n_rows = n_thresholds.min(plane.nrows);
    out.clear();
    for row_idx in 0..n_rows {
        let row = &plane.coeffs[row_idx * x_dim..(row_idx + 1) * x_dim];
        let mut lb_val: f64 = bias_row[row_idx] as f64;
        for d in 0..x_dim {
            let a = row[d] as f64;
            if a >= 0.0 {
                lb_val += a * (clip_l_row[d] as f64);
            } else {
                lb_val += a * (clip_u_row[d] as f64);
            }
        }
        let lb_f32 = if lb_val.is_nan() {
            f32::NEG_INFINITY
        } else {
            ny_tensor::next_down_f32(lb_val as f32)
        };
        out.push((lb_f32, f32::INFINITY));
    }
}

/// Build batched coefficient tensors for one threshold row.
///
/// Extracts `row_idx` from each child's LinearBounds, applying the
/// verify_upper_bound sign convention from `clip_with_precomputed_linear`.
fn build_batched_coefficients(
    linear_bounds_list: &[&LinearBounds],
    row_idx: usize,
    threshold: f32,
    verify_upper_bound: bool,
    n: usize,
    x_dim: usize,
) -> Result<(Array3<f32>, Array2<f32>, Array2<f32>)> {
    let mut l_a_data = Vec::with_capacity(n * x_dim);
    let mut lbias_data = Vec::with_capacity(n);
    let mut thresh_data = Vec::with_capacity(n);

    for (i, lb) in linear_bounds_list.iter().enumerate() {
        let n_rows = lb.lower_a().nrows();
        if row_idx >= n_rows {
            // Row out of range: use zero coefficients (no constraint).
            l_a_data.extend(std::iter::repeat_n(0.0f32, x_dim));
            lbias_data.push(0.0);
            thresh_data.push(0.0);
            continue;
        }

        if verify_upper_bound {
            // upper bound: negate upper_a row and upper_b, negate threshold
            let row = lb.upper_a().row(row_idx);
            if row.len() != x_dim {
                return Err(NyError::InvalidSpec(format!(
                    "batched_clip: child {} upper_a row {} len {} != x_dim {}",
                    i,
                    row_idx,
                    row.len(),
                    x_dim
                )));
            }
            l_a_data.extend(row.iter().map(|v| -v));
            lbias_data.push(-lb.upper_b()[row_idx]);
            thresh_data.push(-threshold);
        } else {
            let row = lb.lower_a().row(row_idx);
            if row.len() != x_dim {
                return Err(NyError::InvalidSpec(format!(
                    "batched_clip: child {} lower_a row {} len {} != x_dim {}",
                    i,
                    row_idx,
                    row.len(),
                    x_dim
                )));
            }
            l_a_data.extend(row.iter().copied());
            lbias_data.push(lb.lower_b()[row_idx]);
            thresh_data.push(threshold);
        }
    }

    let l_a = Array3::from_shape_vec((n, 1, x_dim), l_a_data)
        .map_err(|e| NyError::InvalidSpec(format!("batched_clip: reshape l_a: {}", e)))?;
    let lbias = Array2::from_shape_vec((n, 1), lbias_data)
        .map_err(|e| NyError::InvalidSpec(format!("batched_clip: reshape lbias: {}", e)))?;
    let thresh_mat = Array2::from_shape_vec((n, 1), thresh_data)
        .map_err(|e| NyError::InvalidSpec(format!("batched_clip: reshape thresh: {}", e)))?;

    Ok((l_a, lbias, thresh_mat))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounds::LinearBounds;
    use ndarray::{array, Array1};

    fn make_linear_bounds(
        lower_a: Array2<f32>,
        lower_b: Array1<f32>,
        upper_a: Array2<f32>,
        upper_b: Array1<f32>,
    ) -> LinearBounds {
        LinearBounds::new(lower_a, lower_b, upper_a, upper_b)
            .expect("test linear bounds should be valid")
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_batched_clip_single_child_matches_sequential() {
        // Single child with one threshold: should match clip_with_precomputed_linear.
        let flat_lower = array![0.0, 0.0].into_dyn();
        let flat_upper = array![10.0, 10.0].into_dyn();
        // Constraint: x1 + x2 <= 5, represented as lower_a=[1,1], lower_b=-5
        let lb = make_linear_bounds(
            array![[1.0, 1.0]],
            array![-5.0],
            array![[1.0, 1.0]],
            array![-5.0],
        );

        let result = batched_relaxed_clip_from_flat(
            &[flat_lower],
            &[flat_upper],
            &[&lb],
            &[0.0],
            &[1],
            false,
            1,
        )
        .unwrap();

        assert_eq!(result.clipped_lowers.len(), 1);
        assert_eq!(result.clipped_uppers.len(), 1);
        // Upper bounds should be tightened from 10.0
        assert!(
            result.clipped_uppers[0][[0]] < 10.0,
            "x1 upper should be tightened, got {}",
            result.clipped_uppers[0][[0]]
        );
        assert!(
            result.clipped_uppers[0][[1]] < 10.0,
            "x2 upper should be tightened, got {}",
            result.clipped_uppers[0][[1]]
        );
        // Lower bounds should stay at 0 (positive coefficients only tighten upper)
        assert!(
            (result.clipped_lowers[0][[0]] - 0.0).abs() < 1e-5,
            "x1 lower should stay at 0"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_batched_clip_multiple_children() {
        // Two children with different bounds, same LinearBounds and threshold.
        let flat_lower_1 = array![0.0, 0.0].into_dyn();
        let flat_upper_1 = array![10.0, 10.0].into_dyn();
        let flat_lower_2 = array![0.0, 0.0].into_dyn();
        let flat_upper_2 = array![20.0, 20.0].into_dyn();

        let lb = make_linear_bounds(
            array![[1.0, 1.0]],
            array![-5.0],
            array![[1.0, 1.0]],
            array![-5.0],
        );

        let result = batched_relaxed_clip_from_flat(
            &[flat_lower_1, flat_lower_2],
            &[flat_upper_1, flat_upper_2],
            &[&lb, &lb],
            &[0.0],
            &[1],
            false,
            1,
        )
        .unwrap();

        assert_eq!(result.clipped_lowers.len(), 2);
        // Both children should have tightened upper bounds
        assert!(
            result.clipped_uppers[0][[0]] < 10.0,
            "child 0 x1 upper should be tightened, got {}",
            result.clipped_uppers[0][[0]]
        );
        assert!(
            result.clipped_uppers[1][[0]] < 20.0,
            "child 1 x1 upper should be tightened, got {}",
            result.clipped_uppers[1][[0]]
        );
        // Both lower bounds should stay at 0
        assert!((result.clipped_lowers[0][[0]] - 0.0).abs() < 1e-5);
        assert!((result.clipped_lowers[1][[0]] - 0.0).abs() < 1e-5);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_batched_clip_multiple_thresholds_sequential() {
        // Two thresholds applied sequentially: each should tighten further.
        let flat_lower = array![0.0, 0.0].into_dyn();
        let flat_upper = array![10.0, 10.0].into_dyn();
        // Two constraints:
        // Row 0: x1 <= 5 (represented as lower_a=[1,0], lower_b=-5)
        // Row 1: x2 <= 3 (represented as lower_a=[0,1], lower_b=-3)
        let lb = make_linear_bounds(
            array![[1.0, 0.0], [0.0, 1.0]],
            array![-5.0, -3.0],
            array![[1.0, 0.0], [0.0, 1.0]],
            array![-5.0, -3.0],
        );

        let result = batched_relaxed_clip_from_flat(
            &[flat_lower],
            &[flat_upper],
            &[&lb],
            &[0.0, 0.0],
            &[2],
            false,
            1,
        )
        .unwrap();

        // x1 upper should be ~5, x2 upper should be ~3
        assert!(
            (result.clipped_uppers[0][[0]] - 5.0).abs() < 0.1,
            "x1 upper should be ~5, got {}",
            result.clipped_uppers[0][[0]]
        );
        assert!(
            (result.clipped_uppers[0][[1]] - 3.0).abs() < 0.1,
            "x2 upper should be ~3, got {}",
            result.clipped_uppers[0][[1]]
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_batched_clip_infeasible_marks_verified() {
        // Contradictory constraints should mark child as verified.
        let flat_lower = array![0.0].into_dyn();
        let flat_upper = array![1.0].into_dyn();
        // Row 0: x <= 0.2
        // Row 1: x >= 0.8 -> -x + 0.8 <= 0
        let lb = make_linear_bounds(
            array![[1.0], [-1.0]],
            array![-0.2, 0.8],
            array![[1.0], [-1.0]],
            array![-0.2, 0.8],
        );

        // Single clause (conjunction) of the two contradictory rows -> the
        // within-clause intersection is empty -> verified.
        let result = batched_relaxed_clip_from_flat(
            &[flat_lower],
            &[flat_upper],
            &[&lb],
            &[0.0, 0.0],
            &[2],
            false,
            1,
        )
        .unwrap();

        assert_eq!(result.verified, vec![true]);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_batched_clip_empty_input() {
        let result = batched_relaxed_clip_from_flat(&[], &[], &[], &[0.0], &[1], false, 1).unwrap();
        assert!(result.clipped_lowers.is_empty());
        assert!(result.verified.is_empty());
    }

    // ---- #disj-cross-clause-clip-unsat: clause-aware clip ----

    /// Build LinearBounds whose LOWER plane carries the constraint rows (a_r, b_r);
    /// the upper plane is a copy (unused for the `verify_upper_bound = false`
    /// direction these tests exercise).
    fn lb_from_lower(a: Array2<f32>, b: Array1<f32>) -> LinearBounds {
        make_linear_bounds(a.clone(), b.clone(), a, b)
    }

    /// The exact cross-clause false-verify pattern in 1-D: two single-row clauses
    /// whose feasible half-intervals are DISJOINT ([0, 0.3] and [0.7, 1]). The
    /// historical whole-spec clip intersected them (empty) and wrongly reported
    /// the child verified (a false UNSAT). The clause-aware clip must NOT verify
    /// (each clause is individually feasible) and must carry a box that ENCLOSES
    /// both half-intervals. The SAME two rows as a single conjunctive clause DO
    /// intersect to empty and still verify (unchanged single-clause semantics).
    #[ntest::timeout(10000)]
    #[test]
    fn test_grouped_multiclause_disjoint_clauses_not_verified() {
        let flat_lower = array![0.0].into_dyn();
        let flat_upper = array![1.0].into_dyn();
        // Row 0: x <= 0.3   (a=1, b=0, t=0.3)   -> clause feasible on [0, 0.3]
        // Row 1: -x <= -0.7 (a=-1, b=0, t=-0.7) -> x >= 0.7 -> feasible on [0.7, 1]
        let lb = lb_from_lower(array![[1.0], [-1.0]], array![0.0, 0.0]);
        let thresholds = [0.3f32, -0.7];

        // Two single-row clauses (OR): must NOT verify.
        let res = batched_relaxed_clip_from_flat(
            &[flat_lower.clone()],
            &[flat_upper.clone()],
            &[&lb],
            &thresholds,
            &[1, 1],
            false,
            3,
        )
        .unwrap();
        assert_eq!(
            res.verified,
            vec![false],
            "OR of two individually-feasible clauses must not verify (this is the lsnc false-unsat pattern)"
        );
        // Union box must enclose both [0, 0.3] and [0.7, 1] -> essentially [0, 1].
        assert!(
            res.clipped_lowers[0][[0]] <= 1e-6,
            "union lower must reach 0, got {}",
            res.clipped_lowers[0][[0]]
        );
        assert!(
            res.clipped_uppers[0][[0]] >= 1.0 - 1e-6,
            "union upper must reach 1, got {}",
            res.clipped_uppers[0][[0]]
        );

        // The SAME rows as a single CONJUNCTIVE clause (AND) intersect to empty
        // -> verified (the unchanged single-clause semantics).
        let res_conj = batched_relaxed_clip_from_flat(
            &[flat_lower],
            &[flat_upper],
            &[&lb],
            &thresholds,
            &[2],
            false,
            3,
        )
        .unwrap();
        assert_eq!(
            res_conj.verified,
            vec![true],
            "conjunction x<=0.3 AND x>=0.7 is infeasible -> verified"
        );
    }

    /// Broad enclosure + soundness proptest for the clause-aware clip. Over
    /// random constraint sets, random clause partitions, and random boxes, for
    /// each child:
    ///   * SOUNDNESS: if the clip reports the child verified, NO point strictly
    ///     inside the box satisfies ANY clause's rows;
    ///   * ENCLOSURE: every point that DOES satisfy some clause lies inside the
    ///     carried union box (up to the clip's outward f32 rounding).
    /// Both properties are exactly "no counterexample is discarded / no false
    /// verify" at the geometry the clip operates on (the CROWN lower planes).
    #[ntest::timeout(120000)]
    #[test]
    fn test_grouped_multiclause_enclosure_and_soundness() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC1A5_E5EE_D101);

        let mut saw_verified = false;
        let mut saw_witness_inside = false;
        let mut saw_multiclause = false;

        for _trial in 0..500 {
            let d = rng.random_range(1usize..=3);
            let n_clauses = rng.random_range(1usize..=4);
            let clause_sizes: Vec<usize> = (0..n_clauses)
                .map(|_| rng.random_range(1usize..=3))
                .collect();
            let s: usize = clause_sizes.iter().sum();
            if n_clauses > 1 {
                saw_multiclause = true;
            }

            // Random box (each dim width >= 0.1 so the range is non-empty).
            let mut lo = vec![0f32; d];
            let mut hi = vec![0f32; d];
            for j in 0..d {
                let base = rng.random_range(-2.0f32..2.0);
                let w = rng.random_range(0.1f32..2.0);
                lo[j] = base;
                hi[j] = base + w;
            }
            // Random constraint rows.
            let mut a = Array2::<f32>::zeros((s, d));
            let mut b = Array1::<f32>::zeros(s);
            let mut thr = vec![0f32; s];
            for r in 0..s {
                for j in 0..d {
                    a[[r, j]] = rng.random_range(-2.0f32..2.0);
                }
                b[r] = rng.random_range(-1.0f32..1.0);
                thr[r] = rng.random_range(-1.5f32..1.5);
            }
            let lb = lb_from_lower(a.clone(), b.clone());

            let flat_lower = Array1::from_vec(lo.clone()).into_dyn();
            let flat_upper = Array1::from_vec(hi.clone()).into_dyn();
            let res = batched_relaxed_clip_from_flat(
                &[flat_lower],
                &[flat_upper],
                &[&lb],
                &thr,
                &clause_sizes,
                false,
                3,
            )
            .unwrap();
            let verified = res.verified[0];
            let cl = &res.clipped_lowers[0];
            let cu = &res.clipped_uppers[0];
            if verified {
                saw_verified = true;
            }

            let eps = 1e-3f32; // strict-witness margin
            let tol = 1e-2f32; // enclosure tolerance for outward f32 rounding
            for _sample in 0..600 {
                let mut x = vec![0f32; d];
                for j in 0..d {
                    x[j] = rng.random_range(lo[j]..hi[j]);
                }
                // Does x satisfy some clause (strictly, with margin eps)?
                let mut offset = 0usize;
                let mut satisfies_any = false;
                for &sz in &clause_sizes {
                    let mut clause_ok = true;
                    for r in offset..offset + sz {
                        let mut v = b[r] as f64;
                        for j in 0..d {
                            v += a[[r, j]] as f64 * x[j] as f64;
                        }
                        if v > (thr[r] - eps) as f64 {
                            clause_ok = false;
                            break;
                        }
                    }
                    if clause_ok {
                        satisfies_any = true;
                        break;
                    }
                    offset += sz;
                }
                if satisfies_any {
                    assert!(
                        !verified,
                        "SOUNDNESS: clip reported VERIFIED but a strict witness exists: \
                         x={x:?} box=[{lo:?},{hi:?}] a={a:?} b={b:?} thr={thr:?} clauses={clause_sizes:?}"
                    );
                    for j in 0..d {
                        assert!(
                            x[j] >= cl[[j]] - tol && x[j] <= cu[[j]] + tol,
                            "ENCLOSURE: witness escaped the union box on dim {j}: \
                             x={x:?} clip=[{cl:?},{cu:?}] clauses={clause_sizes:?}"
                        );
                    }
                    saw_witness_inside = true;
                }
            }
        }
        assert!(
            saw_multiclause,
            "fixture must exercise multi-clause configs"
        );
        assert!(saw_verified, "fixture must exercise a verified child");
        assert!(
            saw_witness_inside,
            "fixture must exercise a witness/enclosure check"
        );
    }
}
