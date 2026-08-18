// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::LinearBounds;
use crate::invprop::OutputConstraints;
use ndarray::Array2;
use ny_core::dd::{next_down_f64, next_up_f64};
use ny_core::{f64_to_f32_down, f64_to_f32_up};
use tracing::trace;

/// Add one exact binary32-product term to an outward binary64 interval.
///
/// Both operands have already been promoted from binary32, so their product is
/// exact in binary64 (at most 48 significant bits). Only the reduction add can
/// round; one binary64 step on each side encloses it even under cancellation.
fn add_exact_product_interval(lower: &mut f64, upper: &mut f64, term: f64) {
    *lower = next_down_f64(*lower + term);
    *upper = next_up_f64(*upper + term);
}

/// Store a useful nearest binary32 center plus a symmetric outward error that
/// encloses the complete binary64 interval. Overflow degrades to zero center
/// with infinite error, which the downstream concretizer handles conservatively.
fn coefficient_center_and_error(center: f64, lower: f64, upper: f64) -> (f32, f32) {
    let stored = center as f32;
    if !stored.is_finite() || lower.is_nan() || upper.is_nan() {
        return (0.0, f32::INFINITY);
    }
    let represented = f64::from(stored);
    let lower_gap = next_up_f64((represented - lower).abs());
    let upper_gap = next_up_f64((upper - represented).abs());
    (stored, f64_to_f32_up(next_up_f64(lower_gap.max(upper_gap))))
}

/// Look up gamma `[constraint_idx, output_idx]`, honoring the `share_gammas`
/// (single-column broadcast) layout.
fn gamma_at(gammas: &Array2<f32>, constraint_idx: usize, output_idx: usize) -> f32 {
    if gammas.ncols() == 1 {
        gammas[[constraint_idx, 0]]
    } else if gammas.ncols() > output_idx {
        gammas[[constraint_idx, output_idx]]
    } else {
        0.0
    }
}

/// Whether `bounds` is the untouched output identity seed for `output_dim`:
/// `lower_a == upper_a == I`, `lower_b == upper_b == 0`, no attached error.
///
/// This is the node-identity gate (refutation fix). Folding the raw,
/// output-coordinate-indexed constraint matrix `C` into the coefficient matrix
/// is only *dimensionally valid* at the output seed, where both rows and columns
/// index output coordinates. At any downstream node the columns index that
/// layer's neurons, so folding raw `C` would add mass onto unrelated activations
/// and could INFLATE the bound (a demonstrated false-HOLD path with a hidden
/// layer of width `== output_dim`). Gating on exact identity — rather than the
/// old `num_inputs == output_dim` dimensional coincidence — closes that hole and
/// makes the per-layer / input-level augment call sites sound no-ops.
fn is_output_identity_seed(bounds: &LinearBounds, output_dim: usize) -> bool {
    if bounds.num_outputs() != output_dim || bounds.num_inputs() != output_dim {
        return false;
    }
    if bounds.has_coeff_err() {
        return false;
    }
    let la = bounds.lower_a();
    let ua = bounds.upper_a();
    let lb = bounds.lower_b();
    let ub = bounds.upper_b();
    for i in 0..output_dim {
        if lb[i] != 0.0 || ub[i] != 0.0 {
            return false;
        }
        for j in 0..output_dim {
            let expected = if i == j { 1.0 } else { 0.0 };
            if la[[i, j]] != expected || ua[[i, j]] != expected {
                return false;
            }
        }
    }
    true
}

/// Apply the INVPROP assume-violation output-constraint dual to the output
/// identity seed.
///
/// # Semantics (soundness-critical, settled from the production code path)
///
/// `spec.to_output_constraints()` encodes the **VIOLATION** region
/// `V = {y : C y <= rhs}` as a **conjunction**. INVPROP is an assume-violation
/// prover: assume `C y <= rhs`, dualize with per-output multipliers `gamma >= 0`,
/// and tighten the objective bound. For any `gamma >= 0` and any `y` in `V`
/// (so `C y - rhs <= 0`):
///
/// ```text
/// LOWER:  y_i >= (e_i + C^T gamma_l^(i))^T y  -  gamma_l^(i)^T rhs
/// UPPER:  y_i <= (e_i - C^T gamma_u^(i))^T y  +  gamma_u^(i)^T rhs
/// ```
///
/// Applied to the identity seed (`y = I y + 0`), the merged coefficient row `i`
/// becomes `e_i +/- C^T gamma^(i)` and the constant absorbs `-/+ gamma^(i)^T rhs`.
/// The re-seeded coefficients then propagate through the ordinary CROWN backward,
/// which selects each nonlinearity's relaxation branch by the sign of the
/// *merged* coefficient (**sign-aware concretization**, the interaction
/// refutation fix — never reuse the objective's precomputed branch). The certified
/// per-coefficient error attached here is carried through every linear/conv layer
/// (`Sigma_k err_in . |W|`) and discharged OUTWARD at concretization, so a wrong
/// or suboptimal `gamma` can only LOOSEN the bound, never inflate it.
///
/// # Preconditions / gating (fail-closed)
///
/// - Non-conjunction (disjunction / clause) constraints: returns `bounds`
///   unchanged (a disjunctive violation must never be dualized as one conjunction).
/// - Gamma row count `!= num_constraints`: returns `bounds` unchanged.
/// - `bounds` is not the untouched output identity seed: returns `bounds`
///   unchanged (node-identity gate). This is what makes the historical per-layer
///   and input-level call sites sound no-ops.
///
/// With `gamma == 0` the fold is the identity map (no coefficient delta, no bias
/// delta, no error attached), so enabling INVPROP with un-optimized gammas is
/// byte-identical to the baseline bound.
pub(crate) fn augment_bounds_with_constraints(
    bounds: &LinearBounds,
    constraints: &OutputConstraints,
    gammas_lower: &Array2<f32>,
    gammas_upper: &Array2<f32>,
) -> LinearBounds {
    let num_constraints = constraints.num_constraints();
    let output_dim = constraints.output_dim();

    // Fail-closed: INVPROP dualization is sound only for a conjunctive violation
    // region. A disjunction / clause spec must never reach the dual fold, even if
    // a non-CLI caller reaches this function with active gammas.
    if !constraints.is_conjunction || constraints.clause_indices.is_some() {
        trace!("INVPROP augment: non-conjunction constraints, returning original bounds");
        return bounds.clone();
    }

    // Empty, malformed, or non-finite constraint systems are not valid dual
    // premises. Public fields and serde construction can bypass `new`, so the
    // proof boundary validates the complete matrix here before indexing it.
    if num_constraints == 0
        || output_dim == 0
        || constraints.rhs.len() != num_constraints
        || constraints
            .a_matrix
            .iter()
            .chain(constraints.rhs.iter())
            .any(|value| !value.is_finite())
    {
        trace!("INVPROP augment: invalid constraint system, returning original bounds");
        return bounds.clone();
    }

    let valid_gamma_columns = |columns: usize| columns == 1 || columns == output_dim;
    if gammas_lower.nrows() != num_constraints
        || gammas_upper.nrows() != num_constraints
        || !valid_gamma_columns(gammas_lower.ncols())
        || !valid_gamma_columns(gammas_upper.ncols())
    {
        trace!(
            "INVPROP augment: gamma dimension mismatch (lower {:?}, upper {:?}, constraints {}), \
             returning original bounds",
            gammas_lower.dim(),
            gammas_upper.dim(),
            num_constraints
        );
        return bounds.clone();
    }

    // Dual validity gate. The assume-violation derivation requires gamma >= 0;
    // accepting a malformed negative or non-finite internal value could invert
    // a constraint and manufacture a too-tight bound. Keep the original seed
    // byte-for-byte on any invalid entry.
    if gammas_lower
        .iter()
        .chain(gammas_upper.iter())
        .any(|gamma| !gamma.is_finite() || *gamma < 0.0)
    {
        trace!("INVPROP augment: invalid gamma value, returning original bounds");
        return bounds.clone();
    }

    // Exact-zero gamma (including IEEE -0.0) is mathematically the identity
    // treatment. Return before checking/cloning the identity matrices or
    // entering the O(outputs * constraints * outputs) fold loops. This keeps
    // default-dark INVPROP metadata byte-identical and effectively free.
    if gammas_lower
        .iter()
        .chain(gammas_upper.iter())
        .all(|gamma| *gamma == 0.0)
    {
        return bounds.clone();
    }

    // Node-identity gate — only the untouched output identity seed may fold raw C.
    if !is_output_identity_seed(bounds, output_dim) {
        return bounds.clone();
    }

    let num_outputs = bounds.num_outputs(); // == output_dim at the seed

    let mut lower_a = bounds.lower_a().clone();
    let mut upper_a = bounds.upper_a().clone();
    let mut lower_b = bounds.lower_b().clone();
    let mut upper_b = bounds.upper_b().clone();
    // Certified per-coefficient outward error for the re-seeded coefficients. The
    // augmented coefficient `I +/- C^T gamma` is stored round-to-nearest (the sign
    // of the input it will multiply is unknown here), so soundness rides on this
    // error channel — NOT a directed cast. Both matrices MUST be materialized: a
    // `None` err marks coefficients EXACT and silently skips the concretize penalty.
    let mut lower_a_err = Array2::<f32>::zeros((num_outputs, output_dim));
    let mut upper_a_err = Array2::<f32>::zeros((num_outputs, output_dim));
    let mut any_err = false;
    let mut any_fold_delta = false;

    for i in 0..num_outputs {
        // ---- A-matrix term: w_L = e_i + C^T gamma_l, w_U = e_i - C^T gamma_u ----
        for k in 0..output_dim {
            let mut delta_l = 0.0f64;
            let mut delta_u = 0.0f64;
            let original_l = f64::from(lower_a[[i, k]]);
            let original_u = f64::from(upper_a[[i, k]]);
            let (mut interval_l_lo, mut interval_l_hi) = (original_l, original_l);
            let (mut interval_u_lo, mut interval_u_hi) = (original_u, original_u);
            let mut lower_has_product = false;
            let mut upper_has_product = false;
            for c in 0..num_constraints {
                let ck = constraints.a_matrix[[c, k]] as f64;
                if ck == 0.0 {
                    continue;
                }
                // f32 * f32 promoted to f64 is EXACT; only the running sum rounds.
                let pl = ck * gamma_at(gammas_lower, c, i) as f64;
                let pu = ck * gamma_at(gammas_upper, c, i) as f64;
                delta_l += pl;
                delta_u -= pu; // upper uses -C^T gamma_u
                if pl != 0.0 {
                    lower_has_product = true;
                    add_exact_product_interval(&mut interval_l_lo, &mut interval_l_hi, pl);
                }
                if pu != 0.0 {
                    upper_has_product = true;
                    add_exact_product_interval(&mut interval_u_lo, &mut interval_u_hi, -pu);
                }
            }
            if lower_has_product {
                any_fold_delta = true;
                let (stored, error) = coefficient_center_and_error(
                    original_l + delta_l,
                    interval_l_lo,
                    interval_l_hi,
                );
                lower_a[[i, k]] = stored;
                lower_a_err[[i, k]] = error;
                any_err = true;
            }
            if upper_has_product {
                any_fold_delta = true;
                let (stored, error) = coefficient_center_and_error(
                    original_u + delta_u,
                    interval_u_lo,
                    interval_u_hi,
                );
                upper_a[[i, k]] = stored;
                upper_a_err[[i, k]] = error;
                any_err = true;
            }
        }

        // ---- bias term: lower -= gamma_l.(rhs - C b_L); upper += gamma_u.(rhs - C b_U)
        // The identity admission gate proves b_L == b_U == 0, so C*b is exactly
        // zero. Accumulate the exact f32 products into an outward f64 interval;
        // using only the rounded reduction plus one f32 ULP is not sound under
        // large-small-large cancellation.
        let (mut lower_bias_lo, mut lower_bias_hi) = (0.0f64, 0.0f64);
        let (mut upper_bias_lo, mut upper_bias_hi) = (0.0f64, 0.0f64);
        let mut lower_has_product = false;
        let mut upper_has_product = false;
        for c in 0..num_constraints {
            let rhs = constraints.rhs[c] as f64;
            let lower_term = -(gamma_at(gammas_lower, c, i) as f64 * rhs);
            let upper_term = gamma_at(gammas_upper, c, i) as f64 * rhs;
            if lower_term != 0.0 {
                lower_has_product = true;
                add_exact_product_interval(&mut lower_bias_lo, &mut lower_bias_hi, lower_term);
            }
            if upper_term != 0.0 {
                upper_has_product = true;
                add_exact_product_interval(&mut upper_bias_lo, &mut upper_bias_hi, upper_term);
            }
        }
        // Publish the proof-facing endpoints, never the rounded central sum.
        // The directional converters handle finite binary32 overflow correctly.
        if lower_has_product {
            any_fold_delta = true;
            lower_b[i] = f64_to_f32_down(lower_bias_lo);
        }
        if upper_has_product {
            any_fold_delta = true;
            upper_b[i] = f64_to_f32_up(upper_bias_hi);
        }
    }

    let result = if any_err {
        LinearBounds::new_or_conservative_with_err(
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            lower_a_err,
            upper_a_err,
        )
    } else {
        // Nonzero gamma produced no effective C/rhs delta (for example, an all-zero
        // constraint row), so leave the seed exact with no attached error.
        LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
    };

    match result {
        Ok(augmented) => {
            if any_fold_delta {
                crate::execution_telemetry::record_invprop_nonzero_output_seed_fold();
            }
            augmented
        }
        Err(_) => {
            let (n_out, n_in) = (bounds.num_outputs(), bounds.num_inputs());
            LinearBounds::conservative(n_out, n_in)
        }
    }
}
