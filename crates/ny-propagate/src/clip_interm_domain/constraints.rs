// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Constraint building and preprocessing for intermediate domain clipping.
//!
//! Converts split history into input-space linear constraints and preprocesses
//! them (filtering infeasible/fully-covered, transforming offsets) for the solver.

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};
use std::mem::size_of;

use crate::beta_crown::branching::GraphSplitHistory;
#[cfg(test)]
use crate::beta_crown::constraint_store::{
    ConstraintSense, DomainConstraintStore, LinearConstraintRef,
};

use super::{PreprocessedConstraints, SplitConstraints};

const CLIP_PREPROCESS_MAX_BYTES: usize = 512 * 1024 * 1024;
const CLIP_PREPROCESS_POLL_STRIDE: usize = 1024;
// Conservatively cover each row's ndarray/Vec descriptors and allocator-side
// bookkeeping in addition to explicitly counted payload vectors.
const CLIP_PREPROCESS_ROW_OVERHEAD: usize = 128;

fn check_preprocess_deadline<F>(past_deadline: &mut F, phase: &'static str) -> Result<()>
where
    F: FnMut() -> bool,
{
    if past_deadline() {
        return Err(NyError::DeadlineExceeded(format!(
            "intermediate clipping exceeded deadline during {phase}"
        )));
    }
    Ok(())
}

fn validate_preprocess_allocation(rows: usize, x_dim: usize, copies: usize) -> Result<()> {
    let cells = rows
        .checked_mul(x_dim)
        .ok_or_else(|| NyError::InvalidSpec("clip preprocess shape product overflow".into()))?;
    let matrix_bytes = cells
        .checked_mul(size_of::<f32>())
        .and_then(|n| n.checked_mul(copies))
        .ok_or_else(|| NyError::InvalidSpec("clip preprocess matrix byte overflow".into()))?;
    let row_bytes = rows
        .checked_mul(
            size_of::<Array1<f32>>()
                + size_of::<f32>() * 4
                + size_of::<usize>() * 2
                + size_of::<bool>() * 2
                + CLIP_PREPROCESS_ROW_OVERHEAD,
        )
        .ok_or_else(|| NyError::InvalidSpec("clip preprocess row byte overflow".into()))?;
    let required_bytes = matrix_bytes
        .checked_add(row_bytes)
        .ok_or_else(|| NyError::InvalidSpec("clip preprocess byte sum overflow".into()))?;
    if required_bytes > CLIP_PREPROCESS_MAX_BYTES {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes: CLIP_PREPROCESS_MAX_BYTES,
            site: "clip_interm_preprocess",
        });
    }
    Ok(())
}

/// Build input-space linear constraints from a domain's split history.
///
/// For each split constraint in the history, we use the CROWN linear bounds
/// of the split neuron to create a necessary condition (relaxation) in input space.
/// Rows are materialized directly in normalized `≤` form. The callback-aware
/// implementation keeps deadline polling visible through every dense copy.
///
/// ## Soundness Model
///
/// For a split neuron pre-activation `z(x)` with linear bounds:
/// - lower: `z(x) ≥ lA·x + lbias`
/// - upper: `z(x) ≤ uA·x + ubias`
///
/// We encode split constraints as `A·x + b ≤ 0`:
///
/// - **Inactive branch** (`z(x) ≤ s`): necessary condition is `lA·x + lbias - s ≤ 0`
/// - **Active branch** (`z(x) ≥ s`): necessary condition is `(-uA)·x + (-ubias + s) ≤ 0`
///
/// These are relaxations (necessary but not sufficient conditions), ensuring soundness.
///
/// # Arguments
///
/// * `split_history` - The domain's split history containing all constraints
/// * `linear_bounds_fn` - Function to get CROWN linear bounds for a (node_name, neuron_idx) pair
///   Returns (lA, lbias, uA, ubias) where lA/uA are shape (x_dim,) and biases are scalars.
/// * `x_dim` - Input space dimension
///
/// # Returns
///
/// `SplitConstraints` containing the constraint matrix and vector.
///
/// # References
///
/// - `designs/2026-01-29-clip-interm-domain.md` Section "Soundness model"
/// - `alpha-beta-CROWN/complete_verifier/domain_clipper.py:296`
pub fn build_split_constraints<F>(
    split_history: &GraphSplitHistory,
    linear_bounds_fn: F,
    x_dim: usize,
) -> Result<SplitConstraints>
where
    F: Fn(&str, usize) -> Option<(Array1<f32>, f32, Array1<f32>, f32)>,
{
    let mut never = || false;
    build_split_constraints_with_deadline_check(
        split_history,
        |node, neuron, _deadline: &mut _| linear_bounds_fn(node, neuron),
        x_dim,
        &mut never,
    )
}

pub(crate) fn build_split_constraints_with_deadline_check<F, D>(
    split_history: &GraphSplitHistory,
    linear_bounds_fn: F,
    x_dim: usize,
    past_deadline: &mut D,
) -> Result<SplitConstraints>
where
    F: FnMut(&str, usize, &mut D) -> Option<(Array1<f32>, f32, Array1<f32>, f32)>,
    D: FnMut() -> bool,
{
    let mut linear_bounds_fn = linear_bounds_fn;
    let total_constraints = split_history.depth();
    if total_constraints == 0 {
        return Ok(SplitConstraints::empty(x_dim));
    }

    // Four dense copies cover staging rows, the two callback source rows, and
    // the packed output even in the N=1 boundary case.
    validate_preprocess_allocation(total_constraints, x_dim, 4)?;
    check_preprocess_deadline(past_deadline, "split-constraint allocation")?;
    let mut a_rows = Vec::new();
    let mut b_vals = Vec::new();
    a_rows.try_reserve_exact(total_constraints).map_err(|e| {
        NyError::InvalidSpec(format!("split-constraint row allocation failed: {e}"))
    })?;
    b_vals.try_reserve_exact(total_constraints).map_err(|e| {
        NyError::InvalidSpec(format!("split-constraint bias allocation failed: {e}"))
    })?;

    // Process all constraints (both ReLU and GenBaB)
    let mut cells = 0usize;
    for (constraint_index, constraint) in split_history.iter_all().enumerate() {
        if constraint_index.is_multiple_of(64) {
            check_preprocess_deadline(past_deadline, "split-constraint row")?;
        }
        let node_name = constraint.node_name();
        let neuron_idx = constraint.neuron_idx();
        let split_point = constraint.split_point();
        let is_upper = constraint.is_upper_branch();

        // Get CROWN linear bounds for this neuron
        let bounds = linear_bounds_fn(node_name, neuron_idx, past_deadline);
        // If the callback observed expiry while copying a source row it returns
        // `None`; check immediately so that expiry cannot masquerade as a sound
        // weakening that retains a partial constraint set.
        check_preprocess_deadline(past_deadline, "split-constraint source row")?;
        let Some((l_a, l_bias, u_a, u_bias)) = bounds else {
            // Skip constraints where we don't have linear bounds
            continue;
        };

        if l_a.len() != x_dim || u_a.len() != x_dim {
            return Err(NyError::shape_mismatch(vec![x_dim], l_a.shape().to_vec()));
        }

        // Build the constraint based on branch direction
        let b_val = if is_upper {
            // Active/upper branch: z(x) ≥ split_point
            // Necessary condition: uA·x + ubias ≥ split_point
            // Standard form: (-uA)·x + (-ubias + split_point) ≤ 0
            let Some(b) = add_f32_down(-u_bias, split_point) else {
                // A non-finite source row has no checker-backed meaning. Dropping
                // a necessary condition only weakens the clipping relaxation.
                continue;
            };
            b
        } else {
            // Inactive/lower branch: z(x) ≤ split_point
            // Necessary condition: lA·x + lbias ≤ split_point
            // Standard form: lA·x + (lbias - split_point) ≤ 0
            let Some(b) = sub_f32_down(l_bias, split_point) else {
                continue;
            };
            b
        };

        let mut a_row = Array1::<f32>::zeros(x_dim);
        let mut row_finite = b_val.is_finite();
        for x in 0..x_dim {
            if cells.is_multiple_of(CLIP_PREPROCESS_POLL_STRIDE) {
                check_preprocess_deadline(past_deadline, "split-constraint coefficient")?;
            }
            let value = if is_upper { -u_a[x] } else { l_a[x] };
            row_finite &= value.is_finite();
            a_row[x] = value;
            cells = cells.saturating_add(1);
        }

        // #clip-resnet: on a deep resnet the split neuron's forward linear bound
        // bias can overflow to ±inf in f32 (20+ layers of accumulated relaxation
        // bias). Such a constraint is meaningless (`#2259`). SKIP it — dropping a
        // constraint only WEAKENS the clip's feasible-region shrink, never makes it
        // unsound (the resulting bound is still a valid enclosure). This lets the
        // clip proceed with whatever splits DO have finite linear bounds instead of
        // erroring the whole tightening.
        if !row_finite {
            continue;
        }

        a_rows.push(a_row);
        b_vals.push(b_val);
    }

    check_preprocess_deadline(past_deadline, "split-constraint matrix allocation")?;
    let num_constraints = a_rows.len();
    let mut a_matrix = Array2::zeros((num_constraints, x_dim));
    for (row_index, row) in a_rows.iter().enumerate() {
        for x in 0..x_dim {
            if cells.is_multiple_of(CLIP_PREPROCESS_POLL_STRIDE) {
                check_preprocess_deadline(past_deadline, "split-constraint matrix copy")?;
            }
            a_matrix[[row_index, x]] = row[x];
            cells = cells.saturating_add(1);
        }
    }
    check_preprocess_deadline(past_deadline, "split-constraint completion")?;
    Ok(SplitConstraints {
        a_matrix,
        b_vector: Array1::from_vec(b_vals),
        num_constraints,
    })
}

/// Add two finite `f32` dyadics and store the result toward `-inf`.
///
/// The exact sum of two `f32`s is not in general exactly representable in
/// `f64` (a max-normal plus a subnormal spans far more than 53 significand
/// bits).  We therefore first widen the correctly-rounded `f64` operation one
/// ULP downward, then directionally convert that lower endpoint to `f32`.
/// Positive overflow saturates at `f32::MAX`; negative overflow becomes
/// `-inf`, which is the only `f32` no greater than the exact result.
pub(crate) fn add_f32_down(a: f32, b: f32) -> Option<f32> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    // Preserve the common exact cases instead of needlessly spending one ULP.
    if a == 0.0 {
        return Some(b);
    }
    if b == 0.0 {
        return Some(a);
    }
    if a == -b {
        return Some(0.0);
    }
    let rounded = f64::from(a) + f64::from(b);
    Some(f64_lower_to_f32(f64_next_down(rounded)))
}

pub(crate) fn sub_f32_down(a: f32, b: f32) -> Option<f32> {
    add_f32_down(a, -b)
}

fn f64_lower_to_f32(lower: f64) -> f32 {
    if lower >= f64::from(f32::MAX) {
        return f32::MAX;
    }
    if lower < -f64::from(f32::MAX) {
        return f32::NEG_INFINITY;
    }
    let candidate = lower as f32;
    if f64::from(candidate) <= lower {
        candidate
    } else {
        ny_tensor::next_down_f32(candidate)
    }
}

fn f64_next_down(value: f64) -> f64 {
    if value == f64::NEG_INFINITY || value.is_nan() {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    if value > 0.0 {
        f64::from_bits(value.to_bits() - 1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

#[cfg(test)]
pub(crate) fn split_constraints_from_store(
    store: &DomainConstraintStore,
    x_dim: usize,
) -> Result<SplitConstraints> {
    if store.is_empty() {
        return Ok(SplitConstraints::empty(x_dim));
    }

    let mut a_rows: Vec<Array1<f32>> = Vec::new();
    let mut b_vals: Vec<f32> = Vec::new();

    for constraint in store.iter() {
        let (coeffs, bias) = normalize_constraint(&constraint);

        let mut row = Array1::<f32>::zeros(x_dim);
        for (idx, coeff) in constraint.indices.iter().zip(coeffs.iter()) {
            let col = *idx as usize;
            if col >= x_dim {
                return Err(NyError::InvalidSpec(format!(
                    "constraint index {} out of bounds for x_dim={}",
                    col, x_dim
                )));
            }
            row[col] += *coeff;
        }

        a_rows.push(row);
        b_vals.push(bias);
    }

    let num_constraints = a_rows.len();
    if num_constraints == 0 {
        return Ok(SplitConstraints::empty(x_dim));
    }

    let mut a_matrix = Array2::zeros((num_constraints, x_dim));
    for (i, row) in a_rows.iter().enumerate() {
        a_matrix.row_mut(i).assign(row);
    }

    let b_vector = Array1::from_vec(b_vals);

    Ok(SplitConstraints {
        a_matrix,
        b_vector,
        num_constraints,
    })
}

#[cfg(test)]
fn normalize_constraint(constraint: &LinearConstraintRef<'_>) -> (Vec<f32>, f32) {
    match constraint.sense {
        ConstraintSense::Le => (constraint.coeffs.to_vec(), constraint.bias),
        ConstraintSense::Ge => (
            constraint.coeffs.iter().map(|c| -c).collect(),
            -constraint.bias,
        ),
    }
}

/// Sort and filter constraints for constrained concretization.
///
/// This performs three preprocessing steps:
///
/// 1. **Filter infeasible**: Remove constraints where no point in the box satisfies them
///    (indicates unsatisfiable domain, but we handle gracefully).
///
/// 2. **Filter fully covered**: Remove constraints that are always satisfied by the entire box
///    (they don't affect the optimization).
///
/// 3. **Transform b→d**: Convert offset b to d = A·x0 + b where x0 = (x_L + x_U)/2.
///
/// # Arguments
///
/// * `constraints` - The split constraints to preprocess
/// * `x_l` - Input box lower bounds, shape: `(x_dim,)`
/// * `x_u` - Input box upper bounds, shape: `(x_dim,)`
///
/// # Returns
///
/// `PreprocessedConstraints` with active constraints ready for the solver.
///
/// # References
///
/// - `designs/2026-01-29-clip-interm-domain.md` Section "Preprocess constraints"
/// - `auto_LiRPA/concretize_func.py:sort_out_constr_batches`
pub fn sort_out_constraints(
    constraints: &SplitConstraints,
    x_l: &Array1<f32>,
    x_u: &Array1<f32>,
) -> Result<PreprocessedConstraints> {
    let mut never = || false;
    sort_out_constraints_with_deadline_check(constraints, x_l, x_u, &mut never)
}

pub(crate) fn sort_out_constraints_with_deadline_check<F>(
    constraints: &SplitConstraints,
    x_l: &Array1<f32>,
    x_u: &Array1<f32>,
    past_deadline: &mut F,
) -> Result<PreprocessedConstraints>
where
    F: FnMut() -> bool,
{
    let n = constraints.num_constraints;
    let x_dim = x_l.len();

    if x_u.len() != x_dim
        || constraints.a_matrix.nrows() < n
        || constraints.a_matrix.ncols() != x_dim
        || constraints.b_vector.len() < n
    {
        return Err(NyError::InvalidSpec(format!(
            "clip preprocess shape mismatch: A={:?} b={} n={} x_l={} x_u={}",
            constraints.a_matrix.shape(),
            constraints.b_vector.len(),
            n,
            x_dim,
            x_u.len(),
        )));
    }
    // Original rows, centroid/epsilon, and packed active rows can coexist.
    validate_preprocess_allocation(n, x_dim, 4)?;
    check_preprocess_deadline(past_deadline, "constraint preprocessing")?;

    if n == 0 {
        return Ok(PreprocessedConstraints {
            a_active: Array2::zeros((0, x_dim)),
            b_active: Array1::zeros(0),
            d_active: Array1::zeros(0),
            infeasible_mask: vec![],
            fully_covered_mask: vec![],
        });
    }

    // Compute centroid and half-width with visible polling.
    let mut x0 = Array1::<f32>::zeros(x_dim);
    let mut eps = Array1::<f32>::zeros(x_dim);
    for x in 0..x_dim {
        if x.is_multiple_of(CLIP_PREPROCESS_POLL_STRIDE) {
            check_preprocess_deadline(past_deadline, "constraint preprocessing centroid")?;
        }
        x0[x] = f32::midpoint(x_l[x], x_u[x]);
        eps[x] = (x_u[x] - x_l[x]) / 2.0;
    }

    let mut infeasible_mask = vec![false; n];
    let mut fully_covered_mask = vec![false; n];
    let mut active_indices = Vec::new();
    let mut b_vals = Vec::new();
    let mut d_vals = Vec::new();
    active_indices
        .try_reserve_exact(n)
        .map_err(|e| NyError::InvalidSpec(format!("clip active-index allocation failed: {e}")))?;
    b_vals
        .try_reserve_exact(n)
        .map_err(|e| NyError::InvalidSpec(format!("clip active-bias allocation failed: {e}")))?;
    d_vals
        .try_reserve_exact(n)
        .map_err(|e| NyError::InvalidSpec(format!("clip active-slack allocation failed: {e}")))?;

    let mut cells = 0usize;
    for k in 0..n {
        if k.is_multiple_of(64) {
            check_preprocess_deadline(past_deadline, "constraint preprocessing row")?;
        }
        let a_row = constraints.a_matrix.row(k);
        let b_k = constraints.b_vector[k];

        // Compute d = A·x0 + b
        let mut d_k = b_k;
        let mut max_violation = 0.0f32;
        for x in 0..x_dim {
            if cells.is_multiple_of(CLIP_PREPROCESS_POLL_STRIDE) {
                check_preprocess_deadline(past_deadline, "constraint preprocessing fold")?;
            }
            d_k += a_row[x] * x0[x];
            max_violation += a_row[x].abs() * eps[x];
            cells = cells.saturating_add(1);
        }

        // Check feasibility: if d - max_violation > 0, constraint can never be satisfied
        // (minimum value of A·x + b over the box is positive)
        if d_k - max_violation > 1e-8 {
            infeasible_mask[k] = true;
            continue;
        }

        // Check if fully covered: if d + max_violation <= 0, constraint always satisfied
        // (maximum value of A·x + b over the box is non-positive)
        if d_k + max_violation <= 1e-8 {
            fully_covered_mask[k] = true;
            continue;
        }

        // Active constraint: affects the optimization
        active_indices.push(k);
        b_vals.push(b_k);
        d_vals.push(d_k);
    }

    check_preprocess_deadline(past_deadline, "active-constraint allocation")?;
    let n_active = active_indices.len();
    let mut a_active = Array2::zeros((n_active, x_dim));
    for (i, &source_row) in active_indices.iter().enumerate() {
        for x in 0..x_dim {
            if cells.is_multiple_of(CLIP_PREPROCESS_POLL_STRIDE) {
                check_preprocess_deadline(past_deadline, "active-constraint copy")?;
            }
            a_active[[i, x]] = constraints.a_matrix[[source_row, x]];
            cells = cells.saturating_add(1);
        }
    }

    let d_active = Array1::from_vec(d_vals);
    let b_active = Array1::from_vec(b_vals);

    check_preprocess_deadline(past_deadline, "constraint preprocessing completion")?;
    Ok(PreprocessedConstraints {
        a_active,
        b_active,
        d_active,
        infeasible_mask,
        fully_covered_mask,
    })
}

#[cfg(test)]
mod resource_tests {
    use super::*;

    #[test]
    fn preprocess_allocation_cap_has_checked_exact_boundary() {
        let copies = 4usize;
        let bytes_per_row = copies * size_of::<f32>()
            + size_of::<Array1<f32>>()
            + size_of::<f32>() * 4
            + size_of::<usize>() * 2
            + size_of::<bool>() * 2
            + CLIP_PREPROCESS_ROW_OVERHEAD;
        let last_admitted = CLIP_PREPROCESS_MAX_BYTES / bytes_per_row;
        validate_preprocess_allocation(last_admitted, 1, copies)
            .expect("last byte-budget row must be admitted");
        assert!(matches!(
            validate_preprocess_allocation(last_admitted + 1, 1, copies),
            Err(NyError::CpuMemoryExceeded { .. })
        ));
        assert!(matches!(
            validate_preprocess_allocation(usize::MAX, 2, copies),
            Err(NyError::InvalidSpec(_))
        ));
    }
}
