// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Solver-free sound bounds over a star predicate, via Lagrangian weak duality.
//!
//! [`crate::star_lp`] answers the same questions exactly, but it builds an `ay_milp`
//! session and runs OBBT rounds per call. Measured in the star driver that cost ~30 s PER
//! NODE on a problem with five α variables — solver setup, not search, and it made the
//! exact star path unusable on anything hard.
//!
//! This module trades exactness for speed while keeping soundness absolute.
//!
//! ## The bound
//!
//! For `min (c + g·α)` subject to `A·α ≤ b`, `α ∈ [-1,1]^m`, and any multipliers `λ ≥ 0`:
//!
//! ```text
//!   min (c + g·α)  ≥  L(λ) = c − λᵀb − Σ_j |(g + Aᵀλ)_j|
//! ```
//!
//! because `min_α (v·α)` over the box is `−Σ_j |v_j|`. This is weak duality: EVERY `λ ≥ 0`
//! gives a valid lower bound, so no convergence, no optimality, and no solver correctness is
//! required for the result to be sound. `λ = 0` recovers the plain interval bound exactly,
//! so this can never be worse than the box.
//!
//! Cyclic COORDINATE ASCENT with an exact per-coordinate line search sharpens it, plus a
//! supergradient step to escape a stall (coordinate methods freeze at non-smooth points where
//! only a JOINT move improves). Along one
//! `λ_i` the dual is concave piecewise-linear, so its maximiser is the first breakpoint where
//! the slope turns non-positive — closed form, no step size, no tuning. If the ascent does
//! nothing we still hold a sound bound; it is an optimisation, never a correctness dependency.
//!
//! ## Infeasibility falls out of the same formula
//!
//! Emptiness is the zero objective: if `L(λ) > 0` with `c = 0, g = 0` — i.e.
//! `−λᵀb − Σ_j |(Aᵀλ)_j| > 0` — then `min_α λᵀ(Aα − b) > 0`, so no `α` in the box satisfies
//! `Aα ≤ b`. That is a Farkas-style certificate: finding one PROVES the branch empty, and
//! failing to find one proves nothing (we simply keep the branch).
//!
//! ## Rounding
//!
//! All arithmetic is f64 and the returned interval is widened by a relative epsilon before
//! it is handed back, so floating-point error in the ascent cannot produce a bound that is
//! tighter than the truth.

/// Outward pad sized to the ACTUAL floating-point error of the evaluation.
///
/// The Lagrangian is `c − λᵀb − Σ_j |(g + Aᵀλ)_j|`, i.e. `O(k·m)` multiply-accumulates, each
/// contributing at most one unit of relative rounding error. `terms · EPSILON · magnitude`
/// therefore dominates the true error, with a `+4` cushion for the final subtractions.
///
/// This replaces a flat `1e-9`, which was SEVEN orders of magnitude too conservative and
/// actively harmful: a neuron whose true pre-activation lower bound is exactly `0` came back
/// as `-1.8e-9`, failed the `lo >= 0` stability test, and split — where the same bound from
/// the exact path returned `0.0` and resolved. The pad was manufacturing unstable neurons.
fn outward_pad(lo: f64, hi: f64, terms: usize) -> f64 {
    let magnitude = 1.0 + lo.abs().max(hi.abs());
    #[allow(clippy::cast_precision_loss)]
    let scale = (terms + 4) as f64;
    scale * f64::EPSILON * magnitude
}

/// Widen `lo`/`hi` outward so rounding error cannot narrow the interval.
fn widen_terms(lo: f64, hi: f64, terms: usize) -> (f64, f64) {
    let pad = outward_pad(lo, hi, terms);
    (lo - pad, hi + pad)
}

/// Lower bound of `c + g·α` over `{ A·α ≤ b, α ∈ [-1,1]^m }` by projected dual ascent.
///
/// Returns a SOUND lower bound for any `iters`, including `0`.
fn dual_lower(c: f64, g: &[f64], a_rows: &[Vec<f64>], b: &[f64], iters: usize) -> f64 {
    let k = a_rows.len();
    let mut lambda = vec![0.0f64; k];

    // Objective at the current lambda. Sound for ANY lambda >= 0.
    let eval = |lambda: &[f64]| -> f64 {
        let mut v = g.to_vec();
        for (i, row) in a_rows.iter().enumerate() {
            if lambda[i] == 0.0 {
                continue;
            }
            for (j, &aij) in row.iter().enumerate() {
                v[j] += lambda[i] * aij;
            }
        }
        let abs_sum: f64 = v.iter().map(|x| x.abs()).sum();
        let lam_b: f64 = lambda.iter().zip(b).map(|(l, bi)| l * bi).sum();
        c - lam_b - abs_sum
    };

    let mut best = eval(&lambda);
    if k == 0 {
        return best;
    }
    // Cyclic coordinate ascent with an EXACT line search per coordinate. Each sweep is
    // monotone (a coordinate only moves to its own maximiser), so the value never regresses
    // and stopping early is always safe.
    if iters == 0 {
        return best;
    }
    let sweeps = iters.div_ceil(k).max(1);
    for _ in 0..sweeps {
        let before = best;
        for i in 0..k {
            let keep = lambda[i];
            let t = best_along_coordinate(i, &lambda, g, a_rows, b);
            if !t.is_finite() {
                continue;
            }
            lambda[i] = t;
            let value = eval(&lambda);
            if value.is_finite() && value > best {
                best = value;
            } else {
                lambda[i] = keep;
            }
        }
        if best > before + 1e-12 * (1.0 + best.abs()) {
            continue;
        }
        // STALLED. Coordinate ascent can freeze at a non-smooth point where no SINGLE
        // coordinate improves although a joint move does — e.g. `α <= -0.9 ∧ α >= 0.9`,
        // whose dual grows only along the diagonal λ₁ = λ₂, so each axis alone looks
        // optimal at the origin and the Farkas certificate is never found.
        //
        // Escape with one supergradient step, which moves every coordinate at once. This is
        // the piece the exact line search cannot supply, and the pair covers both regimes:
        // kinks (coordinate) and diagonals (supergradient).
        let mut v = g.to_vec();
        for (i, row) in a_rows.iter().enumerate() {
            if lambda[i] == 0.0 {
                continue;
            }
            for (j, &aij) in row.iter().enumerate() {
                v[j] += lambda[i] * aij;
            }
        }
        let sign: Vec<f64> = v
            .iter()
            .map(|x| {
                if *x > 0.0 {
                    1.0
                } else if *x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect();
        let mut grad = vec![0.0f64; k];
        let mut gnorm = 0.0f64;
        for (i, row) in a_rows.iter().enumerate() {
            let dot: f64 = row.iter().zip(&sign).map(|(a, s)| a * s).sum();
            grad[i] = -b[i] - dot;
            gnorm += grad[i] * grad[i];
        }
        if gnorm <= 0.0 || !gnorm.is_finite() {
            break;
        }
        let step = (1.0 + best.abs()) / gnorm.sqrt();
        let trial: Vec<f64> = lambda
            .iter()
            .zip(&grad)
            .map(|(l, gr)| (l + step * gr).max(0.0))
            .collect();
        if trial.iter().any(|v| !v.is_finite()) {
            break;
        }
        let value = eval(&trial);
        if value.is_finite() && value > best {
            best = value;
            lambda = trial;
        } else {
            break;
        }
    }
    best
}

/// Exact maximiser of `L` along ONE coordinate `λ_i`, holding the rest fixed.
///
/// With `u = g + Σ_{r≠i} λ_r·A_r` and `t = λ_i`:
///
/// ```text
///   L(t) = const − t·b_i − Σ_j |u_j + t·a_ij|
/// ```
///
/// which is concave and PIECEWISE-LINEAR in `t`, with breakpoints at `t_j = −u_j / a_ij`.
/// Its slope `−b_i − Σ_j a_ij·sign(u_j + t·a_ij)` is non-increasing, so the maximiser over
/// `t ≥ 0` is the first breakpoint at which the slope turns non-positive. That is an exact
/// line search in `O(m log m)` — no step size, no tuning, no convergence assumption.
///
/// This is what replaces the subgradient ascent: 64x more subgradient steps bought a 7%
/// reduction in solver calls, because a diminishing `1/t` step cannot find a kink.
fn best_along_coordinate(
    i: usize,
    lambda: &[f64],
    g: &[f64],
    a_rows: &[Vec<f64>],
    b: &[f64],
) -> f64 {
    let m = g.len();
    // u = g + Σ_{r≠i} λ_r·A_r
    let mut u = g.to_vec();
    for (r, row) in a_rows.iter().enumerate() {
        if r == i || lambda[r] == 0.0 {
            continue;
        }
        for (j, &arj) in row.iter().enumerate() {
            u[j] += lambda[r] * arj;
        }
    }
    let ai = &a_rows[i];

    // Candidate breakpoints t >= 0, plus 0 itself.
    let mut pts: Vec<f64> = vec![0.0];
    for j in 0..m {
        if ai[j] != 0.0 {
            let t = -u[j] / ai[j];
            if t.is_finite() && t > 0.0 {
                pts.push(t);
            }
        }
    }
    pts.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    pts.dedup();

    // Slope just to the RIGHT of t: -b_i - Σ_j a_ij·sign(u_j + t·a_ij), evaluated a hair
    // past t so a breakpoint takes its outgoing sign.
    let slope_right = |t: f64| -> f64 {
        let mut acc = -b[i];
        for j in 0..m {
            if ai[j] == 0.0 {
                continue;
            }
            let v = u[j] + t * ai[j];
            let sgn = if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                // At the kink the outgoing branch is decided by a_ij.
                if ai[j] > 0.0 {
                    1.0
                } else {
                    -1.0
                }
            };
            acc -= ai[j] * sgn;
        }
        acc
    };

    // Concave: walk right while the function is still increasing.
    let mut best_t = 0.0;
    for &t in &pts {
        if slope_right(t) > 0.0 {
            // Still climbing past this breakpoint; the optimum is further right.
            continue;
        }
        best_t = t;
        break;
    }
    // Still climbing past the LAST breakpoint: beyond it the slope is constant and
    // positive, so `L` grows without bound along this coordinate. That is not a corner
    // case to clamp away — it is precisely the Farkas signal that the predicate is EMPTY
    // (an unbounded dual certifies primal infeasibility). Ramp `t` out instead of pinning
    // it, and let successive sweeps push it further.
    if pts.iter().all(|&t| slope_right(t) > 0.0) {
        let last = *pts.last().unwrap_or(&0.0);
        best_t = (last + 1.0).max(last * 8.0);
    }
    best_t.max(0.0)
}

/// Sound outward-rounded `(lower, upper)` for `c + g·α` over the star predicate.
///
/// `upper` is obtained as `−min(−c − g·α)`, i.e. the same routine on the negated objective.
///
/// # Panics
/// Never. Malformed inputs (width mismatch) yield the trivial `(-inf, +inf)` interval, which
/// is sound.
#[must_use]
pub fn dual_coordinate_bounds(
    c: f64,
    g: &[f64],
    a_rows: &[Vec<f64>],
    b: &[f64],
    iters: usize,
) -> (f64, f64) {
    if a_rows.len() != b.len() || a_rows.iter().any(|r| r.len() != g.len()) {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    if !c.is_finite() || g.iter().any(|v| !v.is_finite()) {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    let lo = dual_lower(c, g, a_rows, b, iters);
    let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
    let hi = -dual_lower(-c, &neg_g, a_rows, b, iters);
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    let terms = a_rows.len().saturating_mul(g.len()).saturating_add(g.len());
    widen_terms(lo, hi, terms)
}

/// Try to PROVE the star predicate empty.
///
/// `true` is a certificate (a Farkas multiplier was found); `false` means "not proven", not
/// "feasible". Sound in the only direction that matters: a branch is dropped only when it is
/// genuinely unreachable.
#[must_use]
pub fn dual_certifies_empty(a_rows: &[Vec<f64>], b: &[f64], iters: usize) -> bool {
    if a_rows.is_empty() || a_rows.len() != b.len() {
        return false;
    }
    let m = a_rows[0].len();
    if a_rows.iter().any(|r| r.len() != m) {
        return false;
    }
    let zero = vec![0.0f64; m];
    // Strictly positive with a margin, so f64 noise alone cannot manufacture a certificate.
    dual_lower(0.0, &zero, a_rows, b, iters) > 1e-9
}

#[cfg(test)]
#[path = "star_dual_tests.rs"]
mod tests;

/// Evaluate the sound Lagrangian bound at a GIVEN multiplier vector.
///
/// This is the trusted half of the untrusted-solver / trusted-verifier split. `ay_milp`'s
/// float simplex produces a near-optimal `λ` in microseconds; weak duality then makes
///
/// ```text
///   min (c + g·α)  ≥  c − λᵀb − Σ_j |(g + Aᵀλ)_j|
/// ```
///
/// a valid bound for ANY `λ ≥ 0`, however that `λ` was obtained. So the solver never has to
/// be trusted — a wrong `λ` costs tightness, never soundness. Negative entries are clamped
/// to zero, which keeps the formula valid rather than rejecting the vector.
///
/// This is the piece the ascent could not supply: strong duality says a converged dual equals
/// the LP optimum, and a simplex converges where first-order methods crawl.
///
/// Returns `None` on a width mismatch or a non-finite input, so callers fail closed.
#[must_use]
pub fn dual_bound_at(
    c: f64,
    g: &[f64],
    a_rows: &[Vec<f64>],
    b: &[f64],
    lambda: &[f64],
) -> Option<f64> {
    if a_rows.len() != b.len() || lambda.len() != a_rows.len() {
        return None;
    }
    if a_rows.iter().any(|r| r.len() != g.len()) {
        return None;
    }
    if !c.is_finite()
        || g.iter().any(|v| !v.is_finite())
        || b.iter().any(|v| !v.is_finite())
        || lambda.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    let mut v = g.to_vec();
    let mut lam_b = 0.0f64;
    for (i, row) in a_rows.iter().enumerate() {
        let li = lambda[i].max(0.0);
        if li == 0.0 {
            continue;
        }
        lam_b += li * b[i];
        for (j, &aij) in row.iter().enumerate() {
            v[j] += li * aij;
        }
    }
    let abs_sum: f64 = v.iter().map(|x| x.abs()).sum();
    let value = c - lam_b - abs_sum;
    value.is_finite().then_some(value)
}

/// Sound `(lower, upper)` from a solver-supplied `λ` for each direction, outward-rounded.
///
/// `upper` uses the negated objective, exactly as [`dual_coordinate_bounds`] does.
#[must_use]
pub fn dual_bounds_from_multipliers(
    c: f64,
    g: &[f64],
    a_rows: &[Vec<f64>],
    b: &[f64],
    lambda_lo: &[f64],
    lambda_hi: &[f64],
) -> Option<(f64, f64)> {
    let lo = dual_bound_at(c, g, a_rows, b, lambda_lo)?;
    let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
    let hi = -dual_bound_at(-c, &neg_g, a_rows, b, lambda_hi)?;
    let terms = a_rows.len().saturating_mul(g.len()).saturating_add(g.len());
    (lo <= hi).then(|| widen_terms(lo, hi, terms))
}
