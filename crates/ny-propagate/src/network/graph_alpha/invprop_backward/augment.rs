// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::LinearBounds;
use crate::invprop::OutputConstraints;
use ndarray::Array2;
use ny_tensor::{next_down_f32, next_up_f32};
use tracing::trace;

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

    if gammas_lower.nrows() != num_constraints || gammas_upper.nrows() != num_constraints {
        trace!(
            "INVPROP augment: gamma dimension mismatch (lower {:?}, upper {:?}, constraints {}), \
             returning original bounds",
            gammas_lower.dim(),
            gammas_upper.dim(),
            num_constraints
        );
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

    let orig_lower_b = bounds.lower_b();
    let orig_upper_b = bounds.upper_b();

    for i in 0..num_outputs {
        // ---- A-matrix term: w_L = e_i + C^T gamma_l, w_U = e_i - C^T gamma_u ----
        for k in 0..output_dim {
            let mut delta_l = 0.0f64;
            let mut delta_u = 0.0f64;
            let mut abs_l = 0.0f64;
            let mut abs_u = 0.0f64;
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
                abs_l += pl.abs();
                abs_u += pu.abs();
            }
            if delta_l != 0.0 {
                let exact = lower_a[[i, k]] as f64 + delta_l;
                let stored = exact as f32;
                let round_gap = (exact - stored as f64).abs();
                // Certified bound on the f64 accumulation error of `delta_l`
                // (products are exact, so only the sum rounds). Covers arbitrary,
                // possibly-cancelling constraint rows.
                let sum_err = (num_constraints as f64) * f64::EPSILON * abs_l;
                lower_a[[i, k]] = stored;
                lower_a_err[[i, k]] = next_up_f32((round_gap + sum_err) as f32);
                any_err = true;
            }
            if delta_u != 0.0 {
                let exact = upper_a[[i, k]] as f64 + delta_u;
                let stored = exact as f32;
                let round_gap = (exact - stored as f64).abs();
                let sum_err = (num_constraints as f64) * f64::EPSILON * abs_u;
                upper_a[[i, k]] = stored;
                upper_a_err[[i, k]] = next_up_f32((round_gap + sum_err) as f32);
                any_err = true;
            }
        }

        // ---- bias term: lower -= gamma_l.(rhs - C b_L); upper += gamma_u.(rhs - C b_U)
        // `C b_*` is 0 at the identity seed but is computed generally (from the
        // ORIGINAL biases, to avoid aliasing already-mutated rows) so the fold stays
        // correct if this helper is ever reused off-seed.
        let mut lower_bias_delta = 0.0f64;
        let mut upper_bias_delta = 0.0f64;
        for c in 0..num_constraints {
            let mut cb_l = 0.0f64;
            let mut cb_u = 0.0f64;
            for k in 0..output_dim {
                let ck = constraints.a_matrix[[c, k]] as f64;
                if ck != 0.0 {
                    cb_l += ck * orig_lower_b[k] as f64;
                    cb_u += ck * orig_upper_b[k] as f64;
                }
            }
            let rhs = constraints.rhs[c] as f64;
            lower_bias_delta += gamma_at(gammas_lower, c, i) as f64 * (cb_l - rhs);
            upper_bias_delta += gamma_at(gammas_upper, c, i) as f64 * (rhs - cb_u);
        }
        // Directed casts: lower rounds DOWN, upper rounds UP (outward). Skip a zero
        // delta so the f64->f32 round trip cannot nudge an untouched bias by 1 ULP.
        if lower_bias_delta != 0.0 {
            lower_b[i] = next_down_f32((lower_b[i] as f64 + lower_bias_delta) as f32);
        }
        if upper_bias_delta != 0.0 {
            upper_b[i] = next_up_f32((upper_b[i] as f64 + upper_bias_delta) as f32);
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
        // gamma == 0 everywhere: no coefficient delta was stored, so leave the seed
        // exact (no err) — byte-identical to the un-augmented seed.
        LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
    };

    result.unwrap_or_else(|_| {
        let (n_out, n_in) = (bounds.num_outputs(), bounds.num_inputs());
        LinearBounds::conservative(n_out, n_in)
    })
}
