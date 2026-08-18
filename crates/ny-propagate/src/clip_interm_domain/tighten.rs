// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bound tightening via constrained concretization.
//!
//! Uses the complete clipping solver to tighten intermediate bounds
//! subject to split constraints, and provides the high-level integration API.

use ndarray::{Array1, Array2, Array3};
use ny_core::{nan_propagating_max, nan_propagating_min, NyError, Result};
use std::time::Instant;

use crate::beta_crown::branching::GraphSplitHistory;
use crate::complete_clip::{
    check_clip_deadline, complete_clip_certified_with_deadline, validate_clip_work_budget,
    COMPLETE_CLIP_DEFAULT_ITERS,
};

/// Coordinate-ascent iteration count for the constrained-concretization dual
/// solver. Default 1 (COMPLETE_CLIP_DEFAULT_ITERS) is loose; more iterations
/// tighten the dual bound → stronger clip → faster per-subdomain compounding.
/// Cost-only (a tighter valid bound is still sound). Override NY_CLIP_INTERM_ITERS.
fn clip_interm_iters() -> usize {
    std::env::var("NY_CLIP_INTERM_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(COMPLETE_CLIP_DEFAULT_ITERS)
}

use super::constraints::{build_split_constraints, sort_out_constraints};
use super::objectives::select_objective_neurons;
use super::PreprocessedConstraints;

/// Tighten intermediate bounds for selected objective neurons.
///
/// For each objective neuron, uses constrained concretization to compute tighter
/// bounds subject to the split constraints.
///
/// # Arguments
///
/// * `constraints` - Preprocessed split constraints
/// * `objective_bounds` - CROWN linear bounds for objective neurons, shape: `(n_obj, x_dim)` for A, `(n_obj,)` for bias
/// * `x_l` - Input box lower bounds, shape: `(x_dim,)`
/// * `x_u` - Input box upper bounds, shape: `(x_dim,)`
///
/// # Returns
///
/// Tightened bounds as `(lower, upper)` arrays of shape `(n_obj,)`.
///
/// # References
///
/// - `designs/2026-01-29-clip-interm-domain.md` Section "Constrained concretization"
/// - `auto_LiRPA/concretize_func.py:constraints_solving`
pub fn tighten_with_constraints(
    constraints: &PreprocessedConstraints,
    obj_lower_a: &Array2<f32>,
    obj_lower_bias: &Array1<f32>,
    obj_upper_a: &Array2<f32>,
    obj_upper_bias: &Array1<f32>,
    x_l: &Array1<f32>,
    x_u: &Array1<f32>,
) -> Result<(Array1<f32>, Array1<f32>)> {
    tighten_with_constraints_with_deadline(
        constraints,
        obj_lower_a,
        obj_lower_bias,
        obj_upper_a,
        obj_upper_bias,
        x_l,
        x_u,
        None,
    )
}

/// Deadline-aware authority face. Independent dual checking is structural:
/// there is deliberately no environment-controlled unchecked alternative.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tighten_with_constraints_with_deadline(
    constraints: &PreprocessedConstraints,
    obj_lower_a: &Array2<f32>,
    obj_lower_bias: &Array1<f32>,
    obj_upper_a: &Array2<f32>,
    obj_upper_bias: &Array1<f32>,
    x_l: &Array1<f32>,
    x_u: &Array1<f32>,
    deadline: Option<Instant>,
) -> Result<(Array1<f32>, Array1<f32>)> {
    let n_obj = obj_lower_a.nrows();
    let x_dim = x_l.len();
    let n_constraints = constraints.a_active.nrows();
    if x_u.len() != x_dim
        || obj_lower_a.ncols() != x_dim
        || obj_upper_a.shape() != [n_obj, x_dim]
        || obj_lower_bias.len() != n_obj
        || obj_upper_bias.len() != n_obj
        || constraints.a_active.ncols() != x_dim
        || constraints.b_active.len() != n_constraints
    {
        return Err(NyError::InvalidSpec(format!(
            "certified clip shape mismatch: lower_a={:?} upper_a={:?} lower_b={} upper_b={} A={:?} b={} x_l={} x_u={}",
            obj_lower_a.shape(),
            obj_upper_a.shape(),
            obj_lower_bias.len(),
            obj_upper_bias.len(),
            constraints.a_active.shape(),
            constraints.b_active.len(),
            x_dim,
            x_u.len(),
        )));
    }
    if obj_lower_bias.iter().any(|v| !v.is_finite())
        || obj_upper_bias.iter().any(|v| !v.is_finite())
    {
        return Err(NyError::InvalidSpec(
            "certified clip objective biases must be finite".into(),
        ));
    }
    // Preflight before the adapter clones below; the solver repeats the same
    // checked shape products before any of its own allocations.
    validate_clip_work_budget(1, n_obj, n_constraints, x_dim)?;
    check_clip_deadline(deadline, "clip adapter allocations")?;

    // Prepare batched adapter arrays with visible deadline polling rather than
    // opaque ndarray clones. Consume ORIGINAL b: reconstructing it from rounded
    // d=A*x0+b can make the necessary split condition too strict.
    let mut x_l_batched = Array2::<f32>::zeros((1, x_dim));
    check_clip_deadline(deadline, "upper input adapter allocation")?;
    let mut x_u_batched = Array2::<f32>::zeros((1, x_dim));
    check_clip_deadline(deadline, "constraint adapter allocation")?;
    let mut a_matrix_batched = Array3::<f32>::zeros((1, n_constraints, x_dim));
    check_clip_deadline(deadline, "constraint bias adapter allocation")?;
    let mut b_batched = Array2::<f32>::zeros((1, n_constraints));
    let mut cells = 0usize;
    for x in 0..x_dim {
        if cells.is_multiple_of(1024) {
            check_clip_deadline(deadline, "clip input adapter copy")?;
        }
        x_l_batched[[0, x]] = x_l[x];
        x_u_batched[[0, x]] = x_u[x];
        cells = cells.saturating_add(1);
    }
    for k in 0..n_constraints {
        if k.is_multiple_of(64) {
            check_clip_deadline(deadline, "clip constraint adapter row")?;
        }
        b_batched[[0, k]] = constraints.b_active[k];
        for x in 0..x_dim {
            if cells.is_multiple_of(1024) {
                check_clip_deadline(deadline, "clip constraint adapter copy")?;
            }
            a_matrix_batched[[0, k, x]] = constraints.a_active[[k, x]];
            cells = cells.saturating_add(1);
        }
    }
    let x_l_dyn = x_l_batched.into_dyn();
    let x_u_dyn = x_u_batched.into_dyn();
    let a_matrix_dyn = a_matrix_batched.into_dyn();
    let b_dyn = b_batched.into_dyn();

    // Compute tightened lower bounds: min(objective_lower · x) subject to constraints
    let mut objective_lower_batched = Array3::<f32>::zeros((1, n_obj, x_dim));
    for i in 0..n_obj {
        for x in 0..x_dim {
            if cells.is_multiple_of(1024) {
                check_clip_deadline(deadline, "lower objective adapter copy")?;
            }
            objective_lower_batched[[0, i, x]] = obj_lower_a[[i, x]];
            cells = cells.saturating_add(1);
        }
    }
    let objective_lower = objective_lower_batched.into_dyn(); // (1, n_obj, x_dim)
    let lower_result = complete_clip_certified_with_deadline(
        &x_l_dyn,
        &x_u_dyn,
        &objective_lower,
        &a_matrix_dyn,
        &b_dyn,
        -1.0,                // sign=-1 for minimization
        true,                // rearrange constraints
        clip_interm_iters(), // num_iterations (NY_CLIP_INTERM_ITERS override)
        deadline,
    )?;
    drop(objective_lower);

    // Compute tightened upper bounds: max(objective_upper · x) subject to constraints
    check_clip_deadline(deadline, "upper objective adapter allocation")?;
    let mut objective_upper_batched = Array3::<f32>::zeros((1, n_obj, x_dim));
    for i in 0..n_obj {
        for x in 0..x_dim {
            if cells.is_multiple_of(1024) {
                check_clip_deadline(deadline, "upper objective adapter copy")?;
            }
            objective_upper_batched[[0, i, x]] = obj_upper_a[[i, x]];
            cells = cells.saturating_add(1);
        }
    }
    let objective_upper = objective_upper_batched.into_dyn(); // (1, n_obj, x_dim)
    let upper_result = complete_clip_certified_with_deadline(
        &x_l_dyn,
        &x_u_dyn,
        &objective_upper,
        &a_matrix_dyn,
        &b_dyn,
        1.0,                 // sign=+1 for maximization
        true,                // rearrange constraints
        clip_interm_iters(), // num_iterations (NY_CLIP_INTERM_ITERS override)
        deadline,
    )?;
    drop(objective_upper);

    // Extract results and add bias terms
    let lower_raw = lower_result
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| NyError::InvalidSpec(format!("Failed to reshape lower result: {}", e)))?;
    let upper_raw = upper_result
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| NyError::InvalidSpec(format!("Failed to reshape upper result: {}", e)))?;

    check_clip_deadline(deadline, "certified bias merge allocations")?;
    let mut tightened_lower = Array1::zeros(n_obj);
    let mut tightened_upper = Array1::zeros(n_obj);

    for i in 0..n_obj {
        if i.is_multiple_of(1024) {
            check_clip_deadline(deadline, "certified bias merge")?;
        }
        tightened_lower[i] = add_f32_down(lower_raw[[0, i]], obj_lower_bias[i])?;
        tightened_upper[i] = add_f32_up(upper_raw[[0, i]], obj_upper_bias[i])?;
    }

    check_clip_deadline(deadline, "certified clip return")?;
    Ok((tightened_lower, tightened_upper))
}

/// Directed addition for two finite f32 sources. The f64 operation is widened
/// before conversion because a pair of f32 dyadics with a large exponent gap
/// need not have an exact f64 sum. Directional overflow is saturating: the
/// finite-side direction uses `±f32::MAX`, while the unbounded side uses the
/// corresponding infinity.
fn add_f32_down(a: f32, b: f32) -> Result<f32> {
    directed_add_f32(a, b, false)
}

fn add_f32_up(a: f32, b: f32) -> Result<f32> {
    directed_add_f32(a, b, true)
}

fn directed_add_f32(a: f32, b: f32, upward: bool) -> Result<f32> {
    if !a.is_finite() || !b.is_finite() {
        return Err(NyError::InvalidSpec(format!(
            "certified clip bias addition requires finite operands, got {a} and {b}"
        )));
    }
    let rounded = f64::from(a) + f64::from(b);
    let endpoint = if upward {
        next_up_f64(rounded)
    } else {
        next_down_f64(rounded)
    };
    Ok(if upward {
        f64_upper_to_f32(endpoint)
    } else {
        f64_lower_to_f32(endpoint)
    })
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

fn f64_upper_to_f32(upper: f64) -> f32 {
    if upper > f64::from(f32::MAX) {
        return f32::INFINITY;
    }
    if upper <= -f64::from(f32::MAX) {
        return -f32::MAX;
    }
    let candidate = upper as f32;
    if f64::from(candidate) >= upper {
        candidate
    } else {
        ny_tensor::next_up_f32(candidate)
    }
}

fn next_down_f64(value: f64) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude > f64::INFINITY.to_bits() || bits == f64::NEG_INFINITY.to_bits() {
        return value;
    }
    if magnitude == 0 {
        return -f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

fn next_up_f64(value: f64) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude > f64::INFINITY.to_bits() || bits == f64::INFINITY.to_bits() {
        return value;
    }
    if magnitude == 0 {
        return f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// Compute unconstrained bounds using standard interval arithmetic.
#[cfg(test)]
pub(crate) fn compute_unconstrained_bounds(
    obj_lower_a: &Array2<f32>,
    obj_lower_bias: &Array1<f32>,
    obj_upper_a: &Array2<f32>,
    obj_upper_bias: &Array1<f32>,
    x_l: &Array1<f32>,
    x_u: &Array1<f32>,
) -> Result<(Array1<f32>, Array1<f32>)> {
    let n_obj = obj_lower_a.nrows();
    let x_dim = x_l.len();

    let x0 = (x_l + x_u) / 2.0;
    let eps = (x_u - x_l) / 2.0;

    let mut lower = Array1::zeros(n_obj);
    let mut upper = Array1::zeros(n_obj);

    for i in 0..n_obj {
        // Lower bound: A_l·x0 - |A_l|·eps + bias_l
        let mut ax0_l: f32 = 0.0;
        let mut abs_a_eps_l: f32 = 0.0;
        for d in 0..x_dim {
            let a = obj_lower_a[[i, d]];
            ax0_l += a * x0[d];
            abs_a_eps_l += a.abs() * eps[d];
        }
        lower[i] = ax0_l - abs_a_eps_l + obj_lower_bias[i];

        // Upper bound: A_u·x0 + |A_u|·eps + bias_u
        let mut ax0_u: f32 = 0.0;
        let mut abs_a_eps_u: f32 = 0.0;
        for d in 0..x_dim {
            let a = obj_upper_a[[i, d]];
            ax0_u += a * x0[d];
            abs_a_eps_u += a.abs() * eps[d];
        }
        upper[i] = ax0_u + abs_a_eps_u + obj_upper_bias[i];
    }

    Ok((lower, upper))
}

/// High-level API for tightening intermediate bounds using split constraints.
///
/// This is the main entry point for engine integration. Call this after computing
/// CROWN linear bounds and before using intermediate bounds for ReLU relaxation.
///
/// # Arguments
///
/// * `split_history` - The domain's split history
/// * `linear_bounds_for_split` - Function to get linear bounds for split neurons:
///   `(node_name, neuron_idx) -> Option<(lA, lbias, uA, ubias)>`
/// * `linear_bounds_for_objective` - Function to get linear bounds for objective neurons:
///   `(layer_idx, neuron_indices) -> (lA_matrix, lbias, uA_matrix, ubias)`
///   where matrices have shape `(n_neurons, x_dim)` and biases have shape `(n_neurons,)`
/// * `layer_bounds` - Current concrete bounds per layer, each shape `(n_neurons,)`
/// * `input_lower` - Input domain lower bounds
/// * `input_upper` - Input domain upper bounds
/// * `topk` - Number of objective neurons per layer
/// * `coeff_magnitudes` - Optional CROWN coefficient magnitudes per layer for neuron
///   selection. When `Some(&[layer0_mags, layer1_mags, ...])`, neurons are weighted by
///   their CROWN coefficient magnitudes (from `|lA|` or `|uA|`). When `None`, uniform
///   weights are used (all neurons equally weighted). Using actual coefficients improves
///   selection quality by prioritizing neurons with higher bound-tightening impact.
///
/// # Returns
///
/// Updated layer bounds with tightened values for selected neurons.
///
/// # References
///
/// - `designs/2026-01-29-clip-interm-domain.md`
/// - `alpha-beta-CROWN/complete_verifier/domain_clipper.py`
// Justification: Domain clipping needs split history, two closure callbacks (for split/objective
// linear bounds), layer bounds, input bounds, topk, and optional coefficients — all independent.
#[allow(clippy::too_many_arguments)]
pub fn clip_interm_domain_full<FSplit, FObj>(
    split_history: &GraphSplitHistory,
    linear_bounds_for_split: FSplit,
    linear_bounds_for_objective: FObj,
    layer_bounds: &[(Array1<f32>, Array1<f32>)],
    input_lower: &Array1<f32>,
    input_upper: &Array1<f32>,
    topk: usize,
    coeff_magnitudes: Option<&[Array1<f32>]>,
) -> Result<Vec<(Array1<f32>, Array1<f32>)>>
where
    FSplit: Fn(&str, usize) -> Option<(Array1<f32>, f32, Array1<f32>, f32)>,
    FObj: Fn(usize, &[usize]) -> Option<(Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>)>,
{
    let x_dim = input_lower.len();

    // Step 1: Build split constraints
    let constraints = build_split_constraints(split_history, &linear_bounds_for_split, x_dim)?;

    if constraints.is_empty() {
        return Ok(layer_bounds.to_vec());
    }

    // Step 2: Preprocess constraints
    let preprocessed = sort_out_constraints(&constraints, input_lower, input_upper)?;

    if preprocessed.a_active.nrows() == 0 {
        return Ok(layer_bounds.to_vec());
    }

    // Step 3: Tighten bounds for each layer
    let mut result = layer_bounds.to_vec();

    for (layer_idx, (lower, upper)) in layer_bounds.iter().enumerate() {
        // Get coefficient magnitudes for neuron selection
        let n_neurons = lower.len();
        let layer_coeff_magnitudes: std::borrow::Cow<'_, Array1<f32>> = coeff_magnitudes
            .and_then(|cm| cm.get(layer_idx))
            .map(std::borrow::Cow::Borrowed)
            .unwrap_or_else(|| std::borrow::Cow::Owned(Array1::ones(n_neurons)));

        // Select objective neurons
        let selected = select_objective_neurons(lower, upper, &layer_coeff_magnitudes, topk);

        if selected.is_empty() {
            continue;
        }

        // Get linear bounds for selected neurons
        let Some((obj_lower_a, obj_lower_bias, obj_upper_a, obj_upper_bias)) =
            linear_bounds_for_objective(layer_idx, &selected)
        else {
            continue;
        };

        // Tighten bounds
        let (tightened_lower, tightened_upper) = tighten_with_constraints(
            &preprocessed,
            &obj_lower_a,
            &obj_lower_bias,
            &obj_upper_a,
            &obj_upper_bias,
            input_lower,
            input_upper,
        )?;

        // Merge with original bounds
        let (merged_lower, merged_upper) =
            merge_bounds(lower, upper, &tightened_lower, &tightened_upper, &selected);

        result[layer_idx] = (merged_lower, merged_upper);
    }

    Ok(result)
}

/// Merge tightened bounds into existing bounds.
///
/// For each tightened neuron, updates the interval:
/// - `new_l = max(old_l, tightened_l)`
/// - `new_u = min(old_u, tightened_u)`
///
/// If inversion occurs (new_l > new_u), keeps original bounds (numeric fallback).
///
/// # Arguments
///
/// * `original_lower` - Original lower bounds, shape: `(n_neurons,)`
/// * `original_upper` - Original upper bounds, shape: `(n_neurons,)`
/// * `tightened_lower` - Tightened lower bounds for selected neurons
/// * `tightened_upper` - Tightened upper bounds for selected neurons
/// * `selected_indices` - Indices of selected neurons
///
/// # Returns
///
/// Merged bounds as `(lower, upper)` arrays of shape `(n_neurons,)`.
pub fn merge_bounds(
    original_lower: &Array1<f32>,
    original_upper: &Array1<f32>,
    tightened_lower: &Array1<f32>,
    tightened_upper: &Array1<f32>,
    selected_indices: &[usize],
) -> (Array1<f32>, Array1<f32>) {
    let mut lower = original_lower.clone();
    let mut upper = original_upper.clone();

    for (i, &idx) in selected_indices.iter().enumerate() {
        // Use nan_propagating_max/min so upstream NaN is never silently absorbed
        // into a finite bound. IEEE 754 f32::max/min return the non-NaN operand,
        // hiding corruption. See #2858, #2577.
        let new_l = nan_propagating_max(original_lower[idx], tightened_lower[i]);
        let new_u = nan_propagating_min(original_upper[idx], tightened_upper[i]);

        // #3307: this is an intentional keep-original fallback, not a silent repair.
        // The pre-tightening interval is already a sound overapproximation, so if
        // tightening would invert or introduce NaN, keeping the original bounds
        // preserves soundness without discarding valid prior information.
        // Only update if bounds are still valid (no inversion).
        // NaN comparisons return false, so NaN bounds skip the update — the
        // original (pre-tightening) bounds are preserved. This means tightening
        // silently fails for neurons with NaN-tainted CROWN backward results.
        // The NaN is NOT propagated downstream. (#2967)
        if new_l <= new_u {
            lower[idx] = new_l;
            upper[idx] = new_u;
        }
        // If inverted or NaN, keep original bounds (numeric fallback)
    }

    (lower, upper)
}

#[cfg(test)]
mod directed_bias_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn directed_bias_add_handles_subnormal_gap_and_finite_overflow() {
        let tiny = f32::from_bits(1);
        assert!(add_f32_down(1.0, tiny).unwrap() <= 1.0);
        assert!(add_f32_up(1.0, tiny).unwrap() >= ny_tensor::next_up_f32(1.0));

        assert_eq!(add_f32_down(f32::MAX, f32::MAX).unwrap(), f32::MAX);
        assert_eq!(add_f32_up(f32::MAX, f32::MAX).unwrap(), f32::INFINITY);
        assert_eq!(
            add_f32_down(-f32::MAX, -f32::MAX).unwrap(),
            f32::NEG_INFINITY
        );
        assert_eq!(add_f32_up(-f32::MAX, -f32::MAX).unwrap(), -f32::MAX);
    }

    #[test]
    fn authority_face_rejects_nonfinite_objective_bias_before_solving() {
        let constraints = PreprocessedConstraints {
            a_active: Array2::zeros((0, 1)),
            b_active: Array1::zeros(0),
            d_active: Array1::zeros(0),
            infeasible_mask: vec![],
            fully_covered_mask: vec![],
        };
        let err = tighten_with_constraints(
            &constraints,
            &array![[1.0f32]],
            &array![f32::NAN],
            &array![[1.0f32]],
            &array![0.0f32],
            &array![0.0f32],
            &array![1.0f32],
        )
        .expect_err("non-finite bias must fail closed");
        assert!(err.to_string().contains("biases must be finite"));
    }
}
