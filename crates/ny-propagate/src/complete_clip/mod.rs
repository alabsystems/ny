// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Complete Clipping: Coordinate ascent over Lagrange multipliers for near-LP tightness.
//!
//! This implements the Complete Clipping algorithm from Clip-and-Verify, which
//! uses coordinate ascent on the Lagrangian dual to achieve tighter bounds than
//! relaxed clipping.
//!
//! ## Algorithm Overview
//!
//! Given input box `[x_L, x_U]`, CROWN linear constraints `{A_k·x + b_k ≤ 0}`,
//! and objective `c^T x`, solve the Lagrangian dual via coordinate ascent.
//!
//! For each constraint, we find the optimal Lagrange multiplier β_k using
//! closed-form turning point analysis.
//!
//! ## References
//!
//! - Design: `designs/2026-01-28-clip-and-verify-algorithms.md` Section 3
//! - Baseline: `alpha-beta-CROWN/auto_LiRPA/concretize_func.py:constraints_solving`
//! - Paper: Wei et al., "Clip and Verify: Fast and Accurate Neural Network
//!   Verification via Clipping," arXiv:2512.11087

mod affine_provenance;
mod certificate;
mod dual;
pub(crate) mod filter;
mod objective;
mod rearrange;
mod validate;

pub(crate) use affine_provenance::{
    bind_root_sound_crown_rows_to_history, capture_exact_root_input_rows,
    capture_sound_crown_root_rows_at_node, mint_certified_affine_enclosure,
    CertifiedAffineEnclosure, CrownPassStamp, SoundCrownRootAffineRows, ValidatedAffineEnclosure,
};
#[cfg(test)]
pub(crate) use affine_provenance::{
    capture_host_sound_crown_root_rows, check_root_affine_dominance_and_seal,
    test_certified_affine_fixture, TestSoundCrownAffineParts, UntrustedCrownAffineRows,
};

#[cfg(test)]
mod tests;

use ndarray::{s, Array2, Array3, ArrayD, ShapeError};
use ny_core::{NyError, Result};
use std::cmp::Ordering;
use std::mem::size_of;
use std::time::Instant;

use dual::solve_dual_variable_views_with_deadline_check;
use objective::{
    broadcast_objective_with_deadline_check, compute_base_term_with_deadline_check,
    compute_eps_term_with_deadline_check, update_objective_with_deadline_check,
};
use rearrange::{
    compute_constraint_slack_with_deadline_check, rearrange_constraints_with_deadline_check,
};
use validate::ensure_constraints_feasible_with_deadline_check;

/// The coordinate-ascent diagnostic result plus the untrusted dual witness that
/// produced it. Public callers never receive `result`; every authority face
/// independently re-evaluates `beta_store` instead.
struct CompleteClipOutcome {
    result: ArrayD<f32>,
    beta_store: Array3<f32>,
    a_work: Array3<f32>,
    b_work: Array2<f32>,
}

/// Hard host-memory ceiling for one complete-clipping proposal plus its
/// certificate.  This path is a refinement: refusing an oversized proposal
/// leaves the inherited enclosure authoritative.
const COMPLETE_CLIP_MAX_WORK_BYTES: usize = 512 * 1024 * 1024;
const COMPLETE_CLIP_MAX_ARITH_OPS: usize = 500_000_000;
const COMPLETE_CLIP_MAX_ITERS: usize = 64;
/// Conservative budget units charged for each comparison in the turning-point
/// sort.  This covers the comparison itself plus the associated index moves and
/// bookkeeping in Rust's comparison sort.  The logarithmic term is load-bearing:
/// charging only the materialized dense tensors lets a moderate H/N/X shape hide
/// billions of comparisons behind a much smaller linear estimate.
const COMPLETE_CLIP_SORT_OP_WEIGHT: usize = 4;
/// Maximum uninterrupted scalar work between proposal/certificate polls.
pub(super) const CLIP_DEADLINE_POLL_STRIDE: usize = 1024;

/// Check all shape products before any proposal/certificate-owned dense
/// allocation.  Multipliers conservatively cover simultaneously-live ndarray
/// clones and temporaries in the coordinate-ascent implementation.
pub(crate) fn validate_clip_work_budget(
    batch: usize,
    h_dim: usize,
    n_constraints: usize,
    x_dim: usize,
) -> Result<()> {
    fn cells(parts: &[usize]) -> Result<usize> {
        parts.iter().try_fold(1usize, |acc, &n| {
            acc.checked_mul(n)
                .ok_or_else(|| NyError::InvalidSpec("complete clip shape product overflow".into()))
        })
    }
    fn scaled(cells: usize, copies: usize) -> Result<usize> {
        cells
            .checked_mul(copies)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or_else(|| NyError::InvalidSpec("complete clip byte estimate overflow".into()))
    }

    let input = scaled(cells(&[batch, x_dim])?, 8)?;
    let objective = scaled(cells(&[batch, h_dim, x_dim])?, 8)?;
    let constraints = scaled(cells(&[batch, n_constraints, x_dim])?, 5)?;
    let constraint_rows = scaled(cells(&[batch, n_constraints])?, 8)?;
    let witness = scaled(cells(&[batch, h_dim, n_constraints])?, 5)?;
    let output = scaled(cells(&[batch, h_dim])?, 8)?;
    let required_bytes = [
        input,
        objective,
        constraints,
        constraint_rows,
        witness,
        output,
    ]
    .into_iter()
    .try_fold(0usize, |acc, n| acc.checked_add(n))
    .ok_or_else(|| NyError::InvalidSpec("complete clip byte sum overflow".into()))?;
    if required_bytes > COMPLETE_CLIP_MAX_WORK_BYTES {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes: COMPLETE_CLIP_MAX_WORK_BYTES,
            site: "complete_clip_certified",
        });
    }
    // The checker folds every objective coefficient through every constraint.
    // This is not represented by one allocated 4-D tensor, so memory caps alone
    // do not bound runtime (for example, moderate H/N/X can have a huge product).
    let constraint_terms = n_constraints
        .checked_add(1)
        .ok_or_else(|| NyError::InvalidSpec("complete clip operation count overflow".into()))?;
    let arithmetic_ops = [batch, h_dim, x_dim, constraint_terms]
        .into_iter()
        .try_fold(1usize, |acc, n| acc.checked_mul(n))
        .ok_or_else(|| NyError::InvalidSpec("complete clip operation count overflow".into()))?;
    if arithmetic_ops > COMPLETE_CLIP_MAX_ARITH_OPS {
        return Err(NyError::InvalidSpec(format!(
            "complete clip arithmetic budget exceeded: {arithmetic_ops} > {COMPLETE_CLIP_MAX_ARITH_OPS}"
        )));
    }
    Ok(())
}

fn validate_clip_iteration_budget(
    batch: usize,
    h_dim: usize,
    n_constraints: usize,
    x_dim: usize,
    num_iterations: usize,
) -> Result<()> {
    let passes = num_iterations
        .checked_add(1) // independent certificate pass
        .ok_or_else(|| NyError::InvalidSpec("complete clip pass count overflow".into()))?;
    let constraint_terms = n_constraints
        .checked_add(1)
        .ok_or_else(|| NyError::InvalidSpec("complete clip operation count overflow".into()))?;
    let dense_ops = [batch, h_dim, x_dim, constraint_terms, passes]
        .into_iter()
        .try_fold(1usize, |acc, n| acc.checked_mul(n))
        .ok_or_else(|| NyError::InvalidSpec("complete clip operation count overflow".into()))?;
    // `solve_dual_variable` comparison-sorts at most `x_dim` turning points for
    // every (batch, objective, constraint, coordinate pass).  Account for that
    // O(X log X) work explicitly; the previous dense-only estimate admitted the
    // 1x10x4x1,000,000, four-pass case even though it can perform roughly 3.2B
    // comparisons. `sort_levels` is at least one because collection/scanning is
    // still linear for X in {0,1} (also covered by `dense_ops`).
    let sort_levels = if x_dim <= 1 {
        1usize
    } else {
        (usize::BITS - (x_dim - 1).leading_zeros()) as usize
    };
    let sort_ops = [
        batch,
        h_dim,
        n_constraints,
        num_iterations,
        x_dim,
        sort_levels,
        COMPLETE_CLIP_SORT_OP_WEIGHT,
    ]
    .into_iter()
    .try_fold(1usize, |acc, n| acc.checked_mul(n))
    .ok_or_else(|| NyError::InvalidSpec("complete clip sort operation count overflow".into()))?;
    // Optional constraint rearrangement sorts N rows once per batch. Charge it
    // unconditionally so callers cannot accidentally omit the `rearrange`
    // branch from resource authority. The in-place heapsort below performs at
    // most four comparisons per element per ceiling-log level across heap
    // construction and extraction.
    let constraint_sort_levels = if n_constraints <= 1 {
        1usize
    } else {
        (usize::BITS - (n_constraints - 1).leading_zeros()) as usize
    };
    let constraint_sort_ops = [
        batch,
        n_constraints,
        constraint_sort_levels,
        COMPLETE_CLIP_SORT_OP_WEIGHT,
    ]
    .into_iter()
    .try_fold(1usize, |acc, n| acc.checked_mul(n))
    .ok_or_else(|| NyError::InvalidSpec("complete clip constraint sort count overflow".into()))?;
    let arithmetic_ops = dense_ops
        .checked_add(sort_ops)
        .and_then(|n| n.checked_add(constraint_sort_ops))
        .ok_or_else(|| {
            NyError::InvalidSpec("complete clip total operation count overflow".into())
        })?;
    if arithmetic_ops > COMPLETE_CLIP_MAX_ARITH_OPS {
        return Err(NyError::InvalidSpec(format!(
            "complete clip sort-aware iteration budget exceeded: total={arithmetic_ops} \
             (dense={dense_ops}, dual_sort={sort_ops}, constraint_sort={constraint_sort_ops}) \
             > {COMPLETE_CLIP_MAX_ARITH_OPS}"
        )));
    }
    Ok(())
}

pub(super) fn check_clip_deadline_with<F>(past_deadline: &mut F, phase: &'static str) -> Result<()>
where
    F: FnMut() -> bool,
{
    if past_deadline() {
        return Err(NyError::DeadlineExceeded(format!(
            "complete clipping exceeded deadline during {phase}"
        )));
    }
    Ok(())
}

pub(crate) fn check_clip_deadline(deadline: Option<Instant>, phase: &'static str) -> Result<()> {
    let mut past_deadline = || deadline.is_some_and(|d| Instant::now() >= d);
    check_clip_deadline_with(&mut past_deadline, phase)
}

/// Allocation-free heapsort with explicit deadline cancellation and a
/// comparison ceiling charged by `COMPLETE_CLIP_SORT_OP_WEIGHT`.
///
/// Each sift level performs at most two comparisons. Heap construction visits
/// at most `n/2` roots and extraction visits at most `n-1` roots, each with at
/// most `ceil(log2 n)` levels, so `4*n*ceil(log2 n)` is conservative.
pub(crate) fn deadline_heapsort_by<T, F, C>(
    values: &mut [T],
    past_deadline: &mut F,
    phase: &'static str,
    mut compare: C,
) -> Result<()>
where
    F: FnMut() -> bool,
    C: FnMut(&T, &T) -> Ordering,
{
    fn sift_down<T, F, C>(
        values: &mut [T],
        mut root: usize,
        end: usize,
        comparisons: &mut usize,
        past_deadline: &mut F,
        phase: &'static str,
        compare: &mut C,
    ) -> Result<()>
    where
        F: FnMut() -> bool,
        C: FnMut(&T, &T) -> Ordering,
    {
        loop {
            let Some(left) = root.checked_mul(2).and_then(|n| n.checked_add(1)) else {
                return Ok(());
            };
            if left >= end {
                return Ok(());
            }
            let right = left + 1;
            let mut child = left;
            if right < end {
                *comparisons = comparisons.saturating_add(1);
                if comparisons.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                    check_clip_deadline_with(past_deadline, phase)?;
                }
                if compare(&values[left], &values[right]) == Ordering::Less {
                    child = right;
                }
            }
            *comparisons = comparisons.saturating_add(1);
            if comparisons.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                check_clip_deadline_with(past_deadline, phase)?;
            }
            if compare(&values[root], &values[child]) != Ordering::Less {
                return Ok(());
            }
            values.swap(root, child);
            root = child;
        }
    }

    check_clip_deadline_with(past_deadline, phase)?;
    let len = values.len();
    let mut comparisons = 0usize;
    for root in (0..len / 2).rev() {
        sift_down(
            values,
            root,
            len,
            &mut comparisons,
            past_deadline,
            phase,
            &mut compare,
        )?;
    }
    for end in (1..len).rev() {
        values.swap(0, end);
        sift_down(
            values,
            0,
            end,
            &mut comparisons,
            past_deadline,
            phase,
            &mut compare,
        )?;
    }
    check_clip_deadline_with(past_deadline, phase)
}

pub const COMPLETE_CLIP_DEFAULT_ITERS: usize = 1;

/// Convert ndarray ShapeError to NyError
fn shape_err(e: ShapeError) -> NyError {
    NyError::InvalidSpec(format!("array shape error: {}", e))
}

/// Complete Clipping with independently certified Lagrange multipliers.
///
/// Coordinate ascent proposes multipliers; the returned bounds are recomputed
/// from those untrusted multipliers with outward interval arithmetic. A bad
/// multiplier or malformed constraint falls back to the certified box witness.
///
/// # Arguments
///
/// * `x_l` - Lower bounds, shape: `(batch, x_dim)`
/// * `x_u` - Upper bounds, shape: `(batch, x_dim)`
/// * `objective` - Objective coefficients, shape: `(batch, h_dim, x_dim)` or `(h_dim, x_dim)`
/// * `a_matrix` - Constraint coefficients, shape: `(batch, n_constraints, x_dim)`
/// * `b_vector` - Constraint offsets, shape: `(batch, n_constraints)`
/// * `sign` - -1.0 for lower bound (min), +1.0 for upper bound (max)
/// * `rearrange` - Whether to rearrange constraints by distance (default: true)
/// * `num_iterations` - Number of coordinate ascent passes (>= 1)
///
/// # Returns
///
/// Independently certified bound values, shape: `(batch, h_dim)`.
///
/// # Errors
///
/// Returns `NyError::InfeasibleDomain` when any single constraint is infeasible
/// over the input box.
///
/// # Example
///
/// ```text
/// use ny_propagate::complete_clip;
/// use ndarray::{array, Array2, Array3};
///
/// // Single batch, single output dim, 2D input
/// let x_l = array![[0.0, 0.0]];  // shape (1, 2)
/// let x_u = array![[1.0, 1.0]];  // shape (1, 2)
/// let objective = array![[[1.0, 0.0]]]; // shape (1, 1, 2): min x1
/// let a_matrix = array![[[1.0, 1.0]]];  // shape (1, 1, 2): x1 + x2 <= 0.5
/// let b_vector = array![[-0.5]];        // shape (1, 1)
///
/// let result = complete_clip(
///     &x_l.into_dyn(), &x_u.into_dyn(),
///     &objective.into_dyn(),
///     &a_matrix.into_dyn(), &b_vector.into_dyn(),
///     -1.0, true, 1,
/// ).unwrap();
/// ```
///
/// # References
///
/// - `designs/2026-01-28-clip-and-verify-algorithms.md:358`
/// - `alpha-beta-CROWN/auto_LiRPA/concretize_func.py:constraints_solving`
// Justification: Complete CLIP needs input bounds (x_l, x_u), objective vector,
// constraint matrix (A, b), sign, rearrange flag, and iteration count — all independent.
#[allow(clippy::too_many_arguments)]
pub fn complete_clip(
    x_l: &ArrayD<f32>,
    x_u: &ArrayD<f32>,
    objective: &ArrayD<f32>,
    a_matrix: &ArrayD<f32>,
    b_vector: &ArrayD<f32>,
    sign: f32,
    rearrange: bool,
    num_iterations: usize,
) -> Result<ArrayD<f32>> {
    complete_clip_certified_with_deadline(
        x_l,
        x_u,
        objective,
        a_matrix,
        b_vector,
        sign,
        rearrange,
        num_iterations,
        None,
    )
}

/// Complete clipping with an independently checked dual result.
///
/// Coordinate ascent remains a proposal generator: its stored non-negative
/// multipliers are re-evaluated by [`certificate::certify_dual_witness`] using
/// outward interval arithmetic over the exact real values represented by every
/// finite `f32` input.  The checker does not consume the optimizer's reported
/// bound.  Malformed multiplier rows fall back to the independently checked
/// box-only (`beta = 0`) bound.
///
/// Every public and intermediate-domain consumer uses this checked face
/// structurally; no environment setting can downgrade it to the proposal
/// result. This still does not re-authorize the quarantined production clip
/// gates: callers must
/// preserve the CROWN enclosure provenance of the objective and split rows.
/// Deadline-aware certified clipping used by verdict-authoritative refinement
/// lanes. The deadline is checked before every potentially large allocation,
/// each coordinate pass, and throughout independent certificate evaluation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn complete_clip_certified_with_deadline(
    x_l: &ArrayD<f32>,
    x_u: &ArrayD<f32>,
    objective: &ArrayD<f32>,
    a_matrix: &ArrayD<f32>,
    b_vector: &ArrayD<f32>,
    sign: f32,
    rearrange: bool,
    num_iterations: usize,
    deadline: Option<Instant>,
) -> Result<ArrayD<f32>> {
    let mut past_deadline = || deadline.is_some_and(|d| Instant::now() >= d);
    check_clip_deadline_with(&mut past_deadline, "proposal")?;
    let outcome = complete_clip_outcome_with_deadline_check(
        x_l,
        x_u,
        objective,
        a_matrix,
        b_vector,
        sign,
        rearrange,
        num_iterations,
        &mut past_deadline,
    )?;
    check_clip_deadline_with(&mut past_deadline, "certificate")?;
    // Proposal values are diagnostic only. Checking their structural shape
    // makes the field observably consumed without granting any numeric value
    // from coordinate ascent authority over the returned enclosure.
    let expected_shape = [x_l.shape()[0], outcome.beta_store.shape()[1]];
    if outcome.result.shape() != expected_shape {
        return Err(NyError::InternalError(format!(
            "complete clip proposal shape {:?} != certificate shape {:?}",
            outcome.result.shape(),
            expected_shape,
        )));
    }
    let certified = certificate::certify_dual_witness_with_deadline_check(
        x_l,
        x_u,
        objective,
        &outcome.a_work,
        &outcome.b_work,
        &outcome.beta_store,
        sign,
        &mut past_deadline,
    )?;
    // A result that crosses the authority deadline in the certificate tail is
    // still a refusal; no partially or belatedly certified proposal escapes.
    check_clip_deadline_with(&mut past_deadline, "certified return")?;
    Ok(certified)
}

fn copy_constraints_with_deadline_check<F>(
    a_matrix: &ArrayD<f32>,
    b_vector: &ArrayD<f32>,
    past_deadline: &mut F,
) -> Result<(Array3<f32>, Array2<f32>)>
where
    F: FnMut() -> bool,
{
    check_clip_deadline_with(past_deadline, "constraint copy allocation")?;
    let a = a_matrix
        .view()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(shape_err)?;
    let b = b_vector
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(shape_err)?;
    let (batch, n_constraints, x_dim) = (a.shape()[0], a.shape()[1], a.shape()[2]);
    let mut a_copy = Array3::<f32>::zeros((batch, n_constraints, x_dim));
    let mut b_copy = Array2::<f32>::zeros((batch, n_constraints));
    let mut cells = 0usize;
    for batch_idx in 0..batch {
        for k in 0..n_constraints {
            if k.is_multiple_of(64) {
                check_clip_deadline_with(past_deadline, "constraint copy row")?;
            }
            b_copy[[batch_idx, k]] = b[[batch_idx, k]];
            for x in 0..x_dim {
                if cells.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                    check_clip_deadline_with(past_deadline, "constraint coefficient copy")?;
                }
                a_copy[[batch_idx, k, x]] = a[[batch_idx, k, x]];
                cells = cells.saturating_add(1);
            }
        }
    }
    check_clip_deadline_with(past_deadline, "constraint copy completion")?;
    Ok((a_copy, b_copy))
}

fn copy_objective_with_deadline_check<F>(
    objective: &Array3<f32>,
    past_deadline: &mut F,
) -> Result<Array3<f32>>
where
    F: FnMut() -> bool,
{
    check_clip_deadline_with(past_deadline, "objective snapshot allocation")?;
    let shape = objective.shape();
    let mut copy = Array3::<f32>::zeros((shape[0], shape[1], shape[2]));
    for (index, (dst, src)) in copy.iter_mut().zip(objective.iter()).enumerate() {
        if index.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
            check_clip_deadline_with(past_deadline, "objective snapshot copy")?;
        }
        *dst = *src;
    }
    check_clip_deadline_with(past_deadline, "objective snapshot completion")?;
    Ok(copy)
}

#[allow(clippy::too_many_arguments)]
fn complete_clip_outcome_with_deadline_check<F>(
    x_l: &ArrayD<f32>,
    x_u: &ArrayD<f32>,
    objective: &ArrayD<f32>,
    a_matrix: &ArrayD<f32>,
    b_vector: &ArrayD<f32>,
    sign: f32,
    rearrange: bool,
    num_iterations: usize,
    past_deadline: &mut F,
) -> Result<CompleteClipOutcome>
where
    F: FnMut() -> bool,
{
    // Validate shapes
    let x_shape = x_l.shape();
    let x_u_shape = x_u.shape();
    let a_shape = a_matrix.shape();
    let obj_shape = objective.shape();

    if x_shape.len() != 2 {
        return Err(NyError::InvalidSpec(format!(
            "x_l must be 2D (batch, x_dim), got {:?}",
            x_shape
        )));
    }
    if x_u_shape.len() != 2 {
        return Err(NyError::InvalidSpec(format!(
            "x_u must be 2D (batch, x_dim), got {:?}",
            x_u_shape
        )));
    }
    if a_shape.len() != 3 {
        return Err(NyError::InvalidSpec(format!(
            "a_matrix must be 3D (batch, n_constraints, x_dim), got {:?}",
            a_shape
        )));
    }

    let batch = x_shape[0];
    let x_dim = x_shape[1];
    let n_constraints = a_shape[1];

    // Get h_dim from objective (supports both (h_dim, x_dim) and (batch, h_dim, x_dim))
    let h_dim = if obj_shape.len() == 2 {
        obj_shape[0]
    } else if obj_shape.len() == 3 {
        obj_shape[1]
    } else {
        return Err(NyError::InvalidSpec(format!(
            "objective must be 2D or 3D, got {:?}",
            obj_shape
        )));
    };

    // Validate dimensions match
    if a_shape[0] != batch || a_shape[2] != x_dim {
        return Err(NyError::InvalidSpec(format!(
            "a_matrix shape {:?} doesn't match batch={}, x_dim={}",
            a_shape, batch, x_dim
        )));
    }
    if x_u_shape[0] != batch || x_u_shape[1] != x_dim {
        return Err(NyError::InvalidSpec(format!(
            "x_u shape {:?} doesn't match x_l shape {:?}",
            x_u_shape, x_shape
        )));
    }
    if b_vector.ndim() != 2 {
        return Err(NyError::InvalidSpec(format!(
            "b_vector must be 2D (batch, n_constraints), got {:?}",
            b_vector.shape()
        )));
    }
    if b_vector.shape()[0] != batch || b_vector.shape()[1] != n_constraints {
        return Err(NyError::InvalidSpec(format!(
            "b_vector shape {:?} doesn't match batch={}, n_constraints={}",
            b_vector.shape(),
            batch,
            n_constraints
        )));
    }

    if num_iterations == 0 {
        return Err(NyError::InvalidSpec(
            "num_iterations must be >= 1".to_string(),
        ));
    }
    if num_iterations > COMPLETE_CLIP_MAX_ITERS {
        return Err(NyError::InvalidSpec(format!(
            "num_iterations {num_iterations} exceeds certified clip cap {COMPLETE_CLIP_MAX_ITERS}"
        )));
    }
    validate_clip_work_budget(batch, h_dim, n_constraints, x_dim)?;
    validate_clip_iteration_budget(batch, h_dim, n_constraints, x_dim, num_iterations)?;
    check_clip_deadline_with(past_deadline, "proposal allocations")?;
    ensure_constraints_feasible_with_deadline_check(a_matrix, b_vector, x_l, x_u, past_deadline)?;

    // Compute centroid and half-width with the deadline visible inside the
    // dense input loop instead of opaque ndarray arithmetic.
    check_clip_deadline_with(past_deadline, "centroid allocation")?;
    let x_l_2d = x_l
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(shape_err)?;
    let x_u_2d = x_u
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(shape_err)?;
    let mut x0 = Array2::<f32>::zeros((batch, x_dim));
    let mut eps = Array2::<f32>::zeros((batch, x_dim));
    let mut cells = 0usize;
    for b in 0..batch {
        for x in 0..x_dim {
            if cells.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                check_clip_deadline_with(past_deadline, "centroid fold")?;
            }
            x0[[b, x]] = f32::midpoint(x_l_2d[[b, x]], x_u_2d[[b, x]]);
            eps[[b, x]] = (x_u_2d[[b, x]] - x_l_2d[[b, x]]) / 2.0;
            cells = cells.saturating_add(1);
        }
    }
    check_clip_deadline_with(past_deadline, "centroid completion")?;

    // Optionally rearrange constraints by distance to centroid
    let (a_work, b_work) = if rearrange && n_constraints > 1 {
        rearrange_constraints_with_deadline_check(a_matrix, b_vector, x0.view(), past_deadline)?
    } else {
        copy_constraints_with_deadline_check(a_matrix, b_vector, past_deadline)?
    };

    // Pre-compute d = A·x0 + b for each constraint
    // d shape: (batch, n_constraints)
    let d_vector = compute_constraint_slack_with_deadline_check(
        a_work.view(),
        x0.view(),
        b_work.view(),
        past_deadline,
    )?;

    // Broadcast objective to (batch, h_dim, x_dim) if needed
    let mut obj_mat =
        broadcast_objective_with_deadline_check(objective, batch, h_dim, x_dim, past_deadline)?;

    // Convert max to min if needed
    if sign > 0.0 {
        for (index, value) in obj_mat.iter_mut().enumerate() {
            if index.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                check_clip_deadline_with(past_deadline, "objective sign conversion")?;
            }
            *value = -*value;
        }
    }

    // Base term: c^T x0
    // Shape: (batch, h_dim)
    let base_term = compute_base_term_with_deadline_check(&obj_mat, x0.view(), past_deadline)?;
    let obj_initial = copy_objective_with_deadline_check(&obj_mat, past_deadline)?;

    // Dual accumulator: Σ_k β_k * d_k
    // Shape: (batch, h_dim)
    check_clip_deadline_with(past_deadline, "dual accumulator allocation")?;
    let mut dual_part = Array2::<f32>::zeros((batch, h_dim));

    // Track β values for each constraint to enable multiple passes
    // Shape: (batch, h_dim, n_constraints)
    check_clip_deadline_with(past_deadline, "witness allocation")?;
    let mut beta_store = Array3::<f32>::zeros((batch, h_dim, n_constraints));
    check_clip_deadline_with(past_deadline, "coordinate-ascent allocation completion")?;

    // Coordinate ascent: multiple passes over constraints
    for _iter in 0..num_iterations {
        check_clip_deadline_with(past_deadline, "coordinate-ascent pass")?;
        for k in 0..n_constraints {
            if k.is_multiple_of(8) {
                check_clip_deadline_with(past_deadline, "coordinate-ascent row")?;
            }
            let constr_a = a_work.slice(s![.., k, ..]);
            let constr_d = d_vector.slice(s![.., k]);
            let obj_without = {
                let beta_prev = beta_store.slice(s![.., .., k]);
                update_objective_with_deadline_check(
                    &obj_mat,
                    beta_prev,
                    constr_a,
                    -1.0,
                    past_deadline,
                )?
            };

            // Solve for optimal β_k given current obj_mat
            // optimal_beta: (batch, h_dim)
            let optimal_beta = solve_dual_variable_views_with_deadline_check(
                constr_a,
                obj_without.view(),
                constr_d,
                eps.view(),
                past_deadline,
            )?;

            // Update objective: c <- c + β_k * a_k
            obj_mat = update_objective_with_deadline_check(
                &obj_without,
                optimal_beta.view(),
                constr_a,
                1.0,
                past_deadline,
            )?;

            // Accumulate dual part 1: Δβ_k * d_k
            for b in 0..batch {
                for h in 0..h_dim {
                    let cell = b.saturating_mul(h_dim).saturating_add(h);
                    if cell.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                        check_clip_deadline_with(past_deadline, "dual accumulator fold")?;
                    }
                    let previous = beta_store[[b, h, k]];
                    let next = optimal_beta[[b, h]];
                    beta_store[[b, h, k]] = next;
                    dual_part[[b, h]] += (next - previous) * constr_d[b];
                }
            }
        }
    }

    // Final dual part 2: -|c + Σ β_k a_k|^T ε
    // obj_mat: (batch, h_dim, x_dim)
    // eps: (batch, x_dim)
    let final_eps_term = compute_eps_term_with_deadline_check(&obj_mat, eps.view(), past_deadline)?;
    let baseline_eps_term =
        compute_eps_term_with_deadline_check(&obj_initial, eps.view(), past_deadline)?;
    check_clip_deadline_with(past_deadline, "proposal result allocation")?;
    let mut result = Array2::<f32>::zeros((batch, h_dim));
    for b in 0..batch {
        for h in 0..h_dim {
            let cell = b.saturating_mul(h_dim).saturating_add(h);
            if cell.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                check_clip_deadline_with(past_deadline, "proposal result fold")?;
            }
            let baseline_min = base_term[[b, h]] - baseline_eps_term[[b, h]];
            let proposed_min = base_term[[b, h]] + dual_part[[b, h]] - final_eps_term[[b, h]];
            let certified_candidate = if proposed_min.is_nan() || proposed_min < baseline_min {
                baseline_min
            } else {
                proposed_min
            };
            result[[b, h]] = if sign > 0.0 {
                -certified_candidate
            } else {
                certified_candidate
            };
        }
    }
    check_clip_deadline_with(past_deadline, "proposal completion")?;

    Ok(CompleteClipOutcome {
        result: result.into_dyn(),
        beta_store,
        a_work,
        b_work,
    })
}
