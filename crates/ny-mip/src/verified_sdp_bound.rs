// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verified SDP lower bound (Jansson–Chaykin–Keil / VSDP method).
//!
//! This is the semidefinite-programming analogue of NY's existing
//! Neumaier–Shcherbina verified-LP bound: given an *approximate* (possibly
//! infeasible) dual point `y~` produced by any floating-point SDP solver, it
//! produces a **rigorous** lower bound `fL <= p*` on the primal optimum that
//! holds *regardless of solver error*. The floating-point solver is used only
//! as an oracle to propose `y~`; correctness comes entirely from an a-posteriori
//! outward-rounded (interval) verification here on the CPU.
//!
//! # Status: default-off, UNWIRED, moat-safe
//!
//! Nothing outside this module's own `#[cfg(test)]` oracle calls
//! [`verified_sdp_lower_bound`]. It is not reachable from any verifier command
//! or verdict path. Soundness is established by the oracle test in this file
//! (perturbed-dual soundness is the load-bearing assertion). This mirrors the
//! posture of [`crate::certified_box64`]: an independent measurement primitive,
//! separate from the verdict machinery.
//!
//! # The method
//!
//! For the SDP `min <C,X>  s.t. <A_i,X> = b_i,  X >= 0` (dual
//! `max b^T y  s.t.  C - sum_i y_i A_i  >= 0`), and any dual `y~`:
//!
//! 1. **Dual defect** `D = C - sum_i y~_i A_i` (symmetric), computed as an
//!    interval matrix `[D_lo, D_hi]` with outward rounding.
//! 2. A rigorous lower bound `d <= lambda_min(D)` via the tighter of two routes:
//!    - **Gershgorin** (cheap, no factorization):
//!      `d = min_i ( D_ii_lo - sum_{j!=i} |D_ij|_up )`, all outward-rounded.
//!      This mirrors the max-row-abs-sum structure of
//!      `ny-propagate`'s `compute_spectral_norm`.
//!    - **Certified Cholesky** (tighter): a float Cholesky (faer dense LLT) of a
//!      diagonally shifted `mid(D) + s*I` is used *only* to propose a factor `L`;
//!      the bound is then certified a-posteriori by outward-rounded enclosure of
//!      the residual `R = (D + s*I) - L L^T`. Since `L L^T >= 0` exactly (real
//!      `L`) and `||R||_2 <= r` (a rigorously computed max-abs-row-sum, which
//!      upper-bounds the 2-norm of the symmetric `R`), Weyl's inequality gives
//!      `lambda_min(D) >= -s - r`. This needs **no trusted roundoff constant** —
//!      it is a rigorous strengthening of the literal Rump `d = -c` recipe
//!      (there the shift `c` alone bounds the roundoff; here the shift plus the
//!      certified residual do, and a negative shift can even certify a *positive*
//!      lower bound on `lambda_min`).
//! 3. **Assemble** with a primal trace bound `x_bar >= trace(X)` per PSD block:
//!    `fL = b^T y~ + sum_j min(0, d_j) * x_bar_j`, all arithmetic rounded toward
//!    `-inf`. The identity `<C,X> = b^T y + <D,X>` together with
//!    `<D,X> >= lambda_min(D) * trace(X)` for `X >= 0` yields `fL <= p*`.
//!
//! # Scope of this increment
//!
//! Single PSD block. The block-diagonal / multi-block extension is
//! `fL = b^T y~ + sum_j min(0, d_j) * x_bar_j` summed over blocks `j`, each block
//! contributing its own defect sub-matrix, `lambda_min` lower bound `d_j`, and
//! trace bound `x_bar_j`; the per-block machinery here is reused unchanged.
//!
//! # Floating-point assumptions
//!
//! Outward rounding is by the IEEE-754 "nextafter" trick (`f64::next_down` /
//! `f64::next_up`): each directed op steps a full ULP past the round-to-nearest
//! result, which strictly encloses the exact real result. This is sound only
//! with IEEE-754 binary64 gradual underflow (FTZ/DAZ disabled), which is
//! asserted by `require_gradual_underflow` before any bound is trusted; if it
//! does not hold the routine fails closed to `f64::NEG_INFINITY` (a trivially
//! valid lower bound). Any non-finite intermediate likewise fails closed.

use ndarray::Array2;

/// A symmetric `f64` matrix (SDP data block: `C`, each `A_i`, or the defect
/// `D`). Callers pass symmetric matrices; the verified bound reads each entry's
/// interval independently and does not assume exact bit-symmetry of the input.
pub type SymMat = Array2<f64>;

// --------------------------------------------------------------------------
// Local directed-rounding helpers (self-contained; mirror
// `certified_box64::round_down/up/add_down/...` but return `Option` so any
// non-finite result fails the whole bound closed to -inf).
// --------------------------------------------------------------------------

/// Round toward `-inf`: strictly below the exact real value of `x`'s
/// (already round-to-nearest) producing operation. `None` if non-finite.
#[inline]
fn round_dn(x: f64) -> Option<f64> {
    if !x.is_finite() {
        return None;
    }
    let y = x.next_down();
    if y.is_finite() {
        Some(y)
    } else {
        None
    }
}

/// Round toward `+inf`. `None` if non-finite.
#[inline]
fn round_up(x: f64) -> Option<f64> {
    if !x.is_finite() {
        return None;
    }
    let y = x.next_up();
    if y.is_finite() {
        Some(y)
    } else {
        None
    }
}

/// Lower bound on `left + right`.
#[inline]
fn add_dn(left: f64, right: f64) -> Option<f64> {
    if right == 0.0 {
        return Some(left);
    }
    if left == 0.0 {
        return Some(right);
    }
    round_dn(left + right)
}

/// Upper bound on `left + right`.
#[inline]
fn add_up(left: f64, right: f64) -> Option<f64> {
    if right == 0.0 {
        return Some(left);
    }
    if left == 0.0 {
        return Some(right);
    }
    round_up(left + right)
}

/// Lower bound on `left - right`.
#[inline]
fn sub_dn(left: f64, right: f64) -> Option<f64> {
    if right == 0.0 {
        return Some(left);
    }
    if left == 0.0 {
        // negation of a finite value is exact
        return Some(-right);
    }
    round_dn(left - right)
}

/// Upper bound on `left - right`.
#[inline]
fn sub_up(left: f64, right: f64) -> Option<f64> {
    if right == 0.0 {
        return Some(left);
    }
    if left == 0.0 {
        return Some(-right);
    }
    round_up(left - right)
}

/// Lower bound on `left * right`.
#[inline]
fn mul_dn(left: f64, right: f64) -> Option<f64> {
    if left == 0.0 || right == 0.0 {
        return Some(0.0);
    }
    round_dn(left * right)
}

/// Upper bound on `left * right`.
#[inline]
fn mul_up(left: f64, right: f64) -> Option<f64> {
    if left == 0.0 || right == 0.0 {
        return Some(0.0);
    }
    round_up(left * right)
}

/// True iff IEEE-754 binary64 gradual underflow is active (FTZ/DAZ disabled).
/// Mirrors `certified_box64::require_gradual_underflow`; the directed-rounding
/// soundness argument relies on subnormal results not being flushed to zero.
fn require_gradual_underflow() -> bool {
    let half = std::hint::black_box(0.5_f64);
    let min_normal = std::hint::black_box(f64::MIN_POSITIVE);
    let min_subnormal = std::hint::black_box(f64::from_bits(1));
    let two_subnormals = std::hint::black_box(f64::from_bits(2));

    let half_min_normal = std::hint::black_box(min_normal * half);
    let recovered_min_subnormal = std::hint::black_box(two_subnormals * half);
    let added_subnormals = std::hint::black_box(min_subnormal + min_subnormal);
    half_min_normal.to_bits() == 0x0008_0000_0000_0000
        && recovered_min_subnormal.to_bits() == 1
        && added_subnormals.to_bits() == 2
}

// --------------------------------------------------------------------------
// Step 1: dual defect D = C - sum_i y_i A_i, as an outward interval matrix.
// --------------------------------------------------------------------------

/// Outward interval enclosure `[D_lo, D_hi]` of `D = C - sum_i y_i A_i`.
///
/// For a lower bound on each `D_ij` we subtract the *upper* bound of every term
/// `y_i (A_i)_ij` (rounding the running subtraction down); for the upper bound,
/// the *lower* bound of every term (rounding up). The true (real, symmetric)
/// `D` is enclosed entrywise.
fn compute_defect(
    c: &SymMat,
    a: &[SymMat],
    y: &[f64],
    n: usize,
) -> Option<(Array2<f64>, Array2<f64>)> {
    let mut d_lo = Array2::<f64>::zeros((n, n));
    let mut d_hi = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let cij = c[[i, j]];
            if !cij.is_finite() {
                return None;
            }
            let mut lo = cij;
            let mut hi = cij;
            for (ak, &yk) in a.iter().zip(y.iter()) {
                let aij = ak[[i, j]];
                if !aij.is_finite() || !yk.is_finite() {
                    return None;
                }
                let term_lo = mul_dn(yk, aij)?;
                let term_hi = mul_up(yk, aij)?;
                // D -= term : subtract the largest term for the lower bound,
                // the smallest term for the upper bound.
                lo = sub_dn(lo, term_hi)?;
                hi = sub_up(hi, term_lo)?;
            }
            d_lo[[i, j]] = lo;
            d_hi[[i, j]] = hi;
        }
    }
    Some((d_lo, d_hi))
}

// --------------------------------------------------------------------------
// Step 2a: Gershgorin lower bound on lambda_min(D).
// --------------------------------------------------------------------------

/// Rigorous lower bound `d <= lambda_min(D)` from Gershgorin discs.
///
/// For symmetric `D`, `lambda_min(D) >= min_i ( D_ii - sum_{j!=i} |D_ij| )`.
/// We lower-bound each disc left endpoint: `D_ii >= D_ii_lo`, and
/// `sum_{j!=i} |D_ij| <= sum_{j!=i} |D|_up` (up-rounded), so
/// `d = min_i ( D_ii_lo - sum_{j!=i} |D|_up )` (each subtraction rounded down).
fn gershgorin_lambda_min_lb(d_lo: &Array2<f64>, d_hi: &Array2<f64>, n: usize) -> Option<f64> {
    let mut d_min = f64::INFINITY;
    for i in 0..n {
        let mut row_abs_sum = 0.0_f64; // upper bound on sum_{j!=i} |D_ij|
        for j in 0..n {
            if j == i {
                continue;
            }
            // |D_ij| <= max(|D_lo|, |D_hi|) exactly (true value lies in the
            // interval), so no rounding is needed for this magnitude bound.
            let mag = d_lo[[i, j]].abs().max(d_hi[[i, j]].abs());
            if !mag.is_finite() {
                return None;
            }
            row_abs_sum = add_up(row_abs_sum, mag)?;
        }
        let disc = sub_dn(d_lo[[i, i]], row_abs_sum)?;
        if disc < d_min {
            d_min = disc;
        }
    }
    if d_min.is_finite() {
        Some(d_min)
    } else {
        None
    }
}

// --------------------------------------------------------------------------
// Step 2b: certified-Cholesky lower bound on lambda_min(D) (residual-verified).
// --------------------------------------------------------------------------

/// For one diagonal shift `s`, attempt to certify `lambda_min(D) >= -s - r`.
///
/// A float LLT of `mid(D) + s*I` proposes a lower-triangular `L`; the bound is
/// then certified rigorously by outward-rounded enclosure of the symmetric
/// residual `R = (D + s*I) - L L^T` (`r` upper-bounds `||R||_2` via the
/// max-abs-row-sum, valid for symmetric `R`). Returns `None` if the float LLT
/// does not complete, or any enclosure step is non-finite.
fn cholesky_try_shift(d_lo: &Array2<f64>, d_hi: &Array2<f64>, n: usize, s: f64) -> Option<f64> {
    // Float center matrix used ONLY to propose L (correctness is a-posteriori).
    let center = faer::Mat::<f64>::from_fn(n, n, |i, j| {
        let mid = f64::midpoint(d_lo[[i, j]], d_hi[[i, j]]);
        if i == j {
            mid + s
        } else {
            mid
        }
    });
    let chol = center.llt(faer::Side::Lower).ok()?;
    // Owned lower-triangular factor (upper part is zeroed by faer).
    let l = chol.L().to_owned();

    // r = max_i sum_j |R_ij|  >=  ||R||_2  (R symmetric).
    let mut max_row = 0.0_f64;
    for i in 0..n {
        let mut row_sum = 0.0_f64;
        for j in 0..n {
            // Enclose (L L^T)_ij = sum_{k<=min(i,j)} L_ik L_jk.
            let kmax = i.min(j);
            let mut p_lo = 0.0_f64;
            let mut p_hi = 0.0_f64;
            for k in 0..=kmax {
                let lik = l[(i, k)];
                let ljk = l[(j, k)];
                p_lo = add_dn(p_lo, mul_dn(lik, ljk)?)?;
                p_hi = add_up(p_hi, mul_up(lik, ljk)?)?;
            }
            // M_ij interval, with M = D + s*I.
            let (m_lo, m_hi) = if i == j {
                (add_dn(d_lo[[i, i]], s)?, add_up(d_hi[[i, i]], s)?)
            } else {
                (d_lo[[i, j]], d_hi[[i, j]])
            };
            // R_ij in [m_lo - p_hi, m_hi - p_lo].
            let r_lo = sub_dn(m_lo, p_hi)?;
            let r_hi = sub_up(m_hi, p_lo)?;
            let r_abs = r_lo.abs().max(r_hi.abs()); // |R_ij| upper (exact)
            row_sum = add_up(row_sum, r_abs)?;
        }
        if row_sum > max_row {
            max_row = row_sum;
        }
    }
    // lambda_min(D) >= lambda_min(L L^T) + lambda_min(R) - s >= -max_row - s.
    // Round the reported bound down.
    sub_dn(-s, max_row)
}

/// Rigorous lower bound `d <= lambda_min(D)` from the certified-Cholesky route,
/// searching a fixed ladder of diagonal shifts and taking the tightest
/// certifiable bound. Positive shifts help certify a (barely) indefinite `D`;
/// negative shifts (subtracting from the diagonal) can certify a *positive*
/// lower bound when `D` is well-conditioned positive definite. `None` if no
/// shift yields a finite certified bound. `hint` is a suggested positive shift
/// (typically `max(0, -d_gershgorin)`).
fn cholesky_lambda_min_lb(
    d_lo: &Array2<f64>,
    d_hi: &Array2<f64>,
    n: usize,
    hint: f64,
) -> Option<f64> {
    let mut max_diag = 0.0_f64;
    for i in 0..n {
        let v = d_lo[[i, i]].abs().max(d_hi[[i, i]].abs());
        if v.is_finite() && v > max_diag {
            max_diag = v;
        }
    }

    let mut shifts: Vec<f64> = vec![0.0];
    if hint.is_finite() && hint > 0.0 {
        for m in [1.0_f64, 1.000_976_562_5, 2.0, 4.0, 8.0] {
            let s = hint * m;
            if s.is_finite() {
                shifts.push(s);
            }
        }
    }
    if max_diag > 0.0 {
        // Geometric spread of negative shifts as fractions of the largest
        // diagonal magnitude; those that over-shoot lambda_min simply fail.
        for f in [
            0.000_976_562_5,
            0.003_906_25,
            0.015_625,
            0.0625,
            0.25,
            0.5,
            0.75,
            0.9,
            0.99,
            0.999,
            0.9999,
        ] {
            let s = -max_diag * f;
            if s.is_finite() {
                shifts.push(s);
            }
        }
    }

    let mut best = f64::NEG_INFINITY;
    for s in shifts {
        if let Some(cand) = cholesky_try_shift(d_lo, d_hi, n, s) {
            if cand > best {
                best = cand;
            }
        }
    }
    if best.is_finite() {
        Some(best)
    } else {
        None
    }
}

// --------------------------------------------------------------------------
// Public API: assemble the verified lower bound fL <= p*.
// --------------------------------------------------------------------------

/// Rigorous lower bound `fL <= p*` on the SDP primal optimum
/// `p* = min <C,X>  s.t. <A_i,X> = b_i,  X >= 0`, from an approximate dual
/// `y_approx` (single PSD block).
///
/// The bound holds *regardless of solver error* in `y_approx`: a wildly
/// infeasible dual only makes `fL` looser, never unsound. `trace_bounds[0]`
/// must be a valid upper bound `x_bar >= trace(X)` for every feasible `X` (for
/// ReLU-SDP this is finite from the bounded input box).
///
/// Returns [`f64::NEG_INFINITY`] (a trivially valid lower bound) on any
/// dimension mismatch, non-finite input, unsupported floating-point
/// environment, or internal non-finite intermediate — i.e., it fails closed and
/// is never unsound.
///
/// This is an UNWIRED measurement primitive: it is called only by this module's
/// oracle tests and must not be placed on any verdict path.
#[must_use]
pub fn verified_sdp_lower_bound(
    c: &SymMat,
    a: &[SymMat],
    b: &[f64],
    y_approx: &[f64],
    trace_bounds: &[f64],
) -> f64 {
    verified_sdp_lower_bound_inner(c, a, b, y_approx, trace_bounds).unwrap_or(f64::NEG_INFINITY)
}

fn verified_sdp_lower_bound_inner(
    c: &SymMat,
    a: &[SymMat],
    b: &[f64],
    y_approx: &[f64],
    trace_bounds: &[f64],
) -> Option<f64> {
    if !require_gradual_underflow() {
        return None;
    }

    // --- Validate shapes (single PSD block). ---
    let n = c.nrows();
    if n == 0 || c.ncols() != n {
        return None;
    }
    let m = b.len();
    if y_approx.len() != m || a.len() != m {
        return None;
    }
    if trace_bounds.len() != 1 {
        return None;
    }
    for ak in a {
        if ak.nrows() != n || ak.ncols() != n {
            return None;
        }
    }
    let x_bar = trace_bounds[0];
    // trace(X) of a PSD block is nonnegative; the bound must be a nonnegative
    // finite upper bound.
    if !x_bar.is_finite() || x_bar < 0.0 {
        return None;
    }

    // --- Step 1: dual defect interval. ---
    let (d_lo, d_hi) = compute_defect(c, a, y_approx, n)?;

    // --- Step 2: rigorous d <= lambda_min(D), tightest of the two routes. ---
    let d_gersh = gershgorin_lambda_min_lb(&d_lo, &d_hi, n)?;
    let hint = (-d_gersh).max(0.0);
    let d = match cholesky_lambda_min_lb(&d_lo, &d_hi, n, hint) {
        Some(d_chol) => d_gersh.max(d_chol),
        None => d_gersh,
    };

    // --- Step 3: assemble fL = b^T y~ + min(0, d) * x_bar, toward -inf. ---
    let mut bty = 0.0_f64;
    for (&bk, &yk) in b.iter().zip(y_approx.iter()) {
        if !bk.is_finite() || !yk.is_finite() {
            return None;
        }
        let term = mul_dn(bk, yk)?; // lower bound on b_k y_k
        bty = add_dn(bty, term)?;
    }

    // min(0, d) * x_bar : with x_bar >= 0 and min(0,d) <= 0 the correction is
    // <= 0; round it down (toward -inf) for a valid lower bound.
    let correction = if d >= 0.0 { 0.0 } else { mul_dn(d, x_bar)? };

    add_dn(bty, correction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Identity `A_1` (the trace constraint `<I, X> = trace(X) = b_1`).
    fn eye2() -> SymMat {
        arr2(&[[1.0, 0.0], [0.0, 1.0]])
    }

    fn exact_interval(d: &SymMat) -> (Array2<f64>, Array2<f64>) {
        (d.clone(), d.clone())
    }

    // ---- Oracle 1 & 2: exact dual is sound AND tight, on two known SDPs. ----

    // SDP-A: min <C,X> s.t. trace(X)=1, X>=0, C=diag(2,3).
    // Optimal X puts all trace mass on the '2' eigendirection => p* = 2.
    // Strong duality: y* = lambda_min(C) = 2, dual value = b*y* = 2.
    #[test]
    fn exact_dual_diagonal_sdp_is_sound_and_tight() {
        let c = arr2(&[[2.0, 0.0], [0.0, 3.0]]);
        let a = [eye2()];
        let b = [1.0];
        let y_star = [2.0];
        let trace = [1.0]; // trace(X) is pinned to b_1 = 1
        let p_star = 2.0;

        let fl = verified_sdp_lower_bound(&c, &a, &b, &y_star, &trace);
        assert!(fl <= p_star, "fL={fl} must not exceed p*={p_star}");
        assert!(
            (p_star - fl).abs() < 1e-9,
            "fL={fl} should be tight to p*={p_star}"
        );
    }

    // SDP-B: C=[[2,1],[1,2]], trace(X)=1. p* = lambda_min(C) = 1 (eigs 3,1).
    // y* = 1, D = C - I = [[1,1],[1,1]] (PSD, lambda_min = 0).
    #[test]
    fn exact_dual_offdiagonal_sdp_is_sound_and_tight() {
        let c = arr2(&[[2.0, 1.0], [1.0, 2.0]]);
        let a = [eye2()];
        let b = [1.0];
        let y_star = [1.0];
        let trace = [1.0];
        let p_star = 1.0;

        let fl = verified_sdp_lower_bound(&c, &a, &b, &y_star, &trace);
        assert!(fl <= p_star, "fL={fl} must not exceed p*={p_star}");
        assert!(
            (p_star - fl).abs() < 1e-9,
            "fL={fl} should be tight to p*={p_star}"
        );
    }

    // ---- Oracle 3: perturbed dual stays sound (the whole point). ----
    // fL <= p* must hold for ANY y~, feasible or not: the correction term
    // absorbs infeasibility. This is the solver-independence guarantee.
    #[test]
    fn perturbed_dual_is_always_sound() {
        let c = arr2(&[[2.0, 1.0], [1.0, 2.0]]);
        let a = [eye2()];
        let b = [1.0];
        let trace = [1.0];
        let p_star = 1.0;

        for eps in [
            -0.7, -0.3, -0.1, -0.01, -1e-6, 1e-6, 0.01, 0.1, 0.3, 0.7, 1.0, 1.5, 3.0, 10.0,
        ] {
            let y_tilde = [1.0 + eps];
            let fl = verified_sdp_lower_bound(&c, &a, &b, &y_tilde, &trace);
            assert!(fl.is_finite(), "fL must be finite for eps={eps}, got {fl}");
            assert!(
                fl <= p_star,
                "SOUNDNESS VIOLATED: eps={eps} gave fL={fl} > p*={p_star}"
            );
        }
    }

    // Perturbation on the diagonal SDP too, both signs.
    #[test]
    fn perturbed_dual_is_always_sound_diagonal() {
        let c = arr2(&[[2.0, 0.0], [0.0, 3.0]]);
        let a = [eye2()];
        let b = [1.0];
        let trace = [1.0];
        let p_star = 2.0;

        for eps in [-1.0, -0.5, -0.05, 0.05, 0.5, 1.0, 2.5, 5.0] {
            let y_tilde = [2.0 + eps];
            let fl = verified_sdp_lower_bound(&c, &a, &b, &y_tilde, &trace);
            assert!(fl.is_finite());
            assert!(
                fl <= p_star,
                "SOUNDNESS VIOLATED: eps={eps} gave fL={fl} > p*={p_star}"
            );
        }
    }

    // ---- Oracle 4: non-PSD defect still yields a valid (looser) bound. ----
    #[test]
    fn indefinite_defect_still_valid_lower_bound() {
        let c = arr2(&[[2.0, 1.0], [1.0, 2.0]]);
        let a = [eye2()];
        let b = [1.0];
        let trace = [1.0];
        let p_star = 1.0;

        // y~ = 5: D = C - 5I = [[-3,1],[1,-3]] is negative definite (indefinite
        // vs 0). The defect-correction term keeps fL a valid lower bound.
        let fl_bad = verified_sdp_lower_bound(&c, &a, &b, &[5.0], &trace);
        assert!(fl_bad.is_finite());
        assert!(fl_bad <= p_star, "fL_bad={fl_bad} must not exceed p*");

        // y~ = 0.5 (dual-feasible but far from optimal): D=[[1.5,1],[1,1.5]] is
        // PSD so the correction is 0 and fL = b*y~ ~ 0.5 -- a valid, LOOSE bound.
        let fl_loose = verified_sdp_lower_bound(&c, &a, &b, &[0.5], &trace);
        assert!(fl_loose <= p_star, "fL_loose={fl_loose} must not exceed p*");
        assert!(
            fl_loose < 0.9,
            "fL_loose={fl_loose} should be a demonstrably looser bound"
        );
    }

    // ---- Oracle 5: determinism + outward-rounding direction sanity. ----
    #[test]
    fn deterministic_and_below_naive_bty() {
        let c = arr2(&[[2.0, 1.0], [1.0, 2.0]]);
        let a = [eye2()];
        let b = [1.0];
        let y = [1.0];
        let trace = [1.0];

        let fl1 = verified_sdp_lower_bound(&c, &a, &b, &y, &trace);
        let fl2 = verified_sdp_lower_bound(&c, &a, &b, &y, &trace);
        assert_eq!(fl1.to_bits(), fl2.to_bits(), "fL must be bit-deterministic");

        // fL must never exceed the naive (round-to-nearest) dual value b^T y~.
        let naive_bty: f64 = b.iter().zip(y.iter()).map(|(bk, yk)| bk * yk).sum();
        assert!(
            fl1 <= naive_bty,
            "fL={fl1} must be <= naive b^T y~={naive_bty}"
        );
    }

    // ---- Direct eigenvalue-route tests (the certificates behind fL). ----

    #[test]
    fn gershgorin_lower_bounds_lambda_min() {
        // D=[[2,1],[1,2]]: lambda_min = 1; Gershgorin disc = 2 - 1 = 1 (tight).
        let d = arr2(&[[2.0, 1.0], [1.0, 2.0]]);
        let (lo, hi) = exact_interval(&d);
        let g = gershgorin_lambda_min_lb(&lo, &hi, 2).unwrap();
        assert!(g <= 1.0 + 1e-12, "g={g} must lower-bound lambda_min=1");
        assert!(g > 1.0 - 1e-9, "g={g} should be near-tight (=1)");

        // D=diag(0,1): lambda_min = 0.
        let d2 = arr2(&[[0.0, 0.0], [0.0, 1.0]]);
        let (lo2, hi2) = exact_interval(&d2);
        let g2 = gershgorin_lambda_min_lb(&lo2, &hi2, 2).unwrap();
        assert!(
            g2 <= 0.0 + 1e-12 && g2 > -1e-9,
            "g2={g2} must ~lower-bound 0"
        );
    }

    #[test]
    fn cholesky_route_beats_gershgorin_on_dominant_offdiagonal() {
        // D=[[2,1,1],[1,2,1],[1,1,2]] : eigenvalues {4,1,1}, lambda_min = 1.
        // Gershgorin disc = 2 - (1+1) = 0 (loose). Certified Cholesky, via a
        // negative diagonal shift, certifies a positive lower bound < 1.
        let d = arr2(&[[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]);
        let (lo, hi) = exact_interval(&d);

        let g = gershgorin_lambda_min_lb(&lo, &hi, 3).unwrap();
        assert!(g <= 1.0, "Gershgorin g={g} must lower-bound lambda_min=1");
        assert!(g < 0.01, "Gershgorin should be loose (~0) here, got {g}");

        let hint = (-g).max(0.0);
        let ch = cholesky_lambda_min_lb(&lo, &hi, 3, hint).unwrap();
        // Sound: never exceed the true lambda_min = 1.
        assert!(ch <= 1.0 + 1e-12, "Cholesky ch={ch} must lower-bound 1");
        // Value-add: strictly tighter than the (loose) Gershgorin bound.
        assert!(
            ch > g + 0.1,
            "Cholesky ch={ch} should beat Gershgorin g={g}"
        );
        assert!(
            ch > 0.0,
            "Cholesky should certify a positive bound, got {ch}"
        );
    }

    #[test]
    fn cholesky_route_is_sound_on_indefinite_matrix() {
        // D=[[-3,1],[1,-3]] : eigenvalues {-2,-4}, lambda_min = -4. Any
        // certified Cholesky bound must stay <= -4.
        let d = arr2(&[[-3.0, 1.0], [1.0, -3.0]]);
        let (lo, hi) = exact_interval(&d);
        let g = gershgorin_lambda_min_lb(&lo, &hi, 2).unwrap();
        assert!(g <= -4.0 + 1e-9, "g={g} must lower-bound lambda_min=-4");
        let hint = (-g).max(0.0);
        if let Some(ch) = cholesky_lambda_min_lb(&lo, &hi, 2, hint) {
            assert!(ch <= -4.0 + 1e-9, "Cholesky ch={ch} must lower-bound -4");
        }
    }

    // ---- Defect interval sanity: outward enclosure of exact D. ----
    #[test]
    fn defect_interval_encloses_exact_defect() {
        // C=[[2,1],[1,2]], A=I, y=1 => D = [[1,1],[1,1]] exactly.
        let c = arr2(&[[2.0, 1.0], [1.0, 2.0]]);
        let a = [eye2()];
        let y = [1.0];
        let (lo, hi) = compute_defect(&c, &a, &y, 2).unwrap();
        let exact = 1.0;
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    lo[[i, j]] <= exact && exact <= hi[[i, j]],
                    "entry ({i},{j}): [{},{}] must enclose {exact}",
                    lo[[i, j]],
                    hi[[i, j]]
                );
            }
        }
    }

    // ---- Fail-closed: malformed input returns -inf, never a bogus number. ----
    #[test]
    fn malformed_input_fails_closed_to_neg_infinity() {
        let c = arr2(&[[2.0, 1.0], [1.0, 2.0]]);
        let a = [eye2()];
        // Wrong trace_bounds length (single-block requires exactly 1).
        assert_eq!(
            verified_sdp_lower_bound(&c, &a, &[1.0], &[1.0], &[1.0, 2.0]),
            f64::NEG_INFINITY
        );
        // Mismatched dual length.
        assert_eq!(
            verified_sdp_lower_bound(&c, &a, &[1.0], &[1.0, 2.0], &[1.0]),
            f64::NEG_INFINITY
        );
        // Non-finite input.
        let bad_c = arr2(&[[f64::NAN, 0.0], [0.0, 1.0]]);
        assert_eq!(
            verified_sdp_lower_bound(&bad_c, &a, &[1.0], &[1.0], &[1.0]),
            f64::NEG_INFINITY
        );
    }
}
