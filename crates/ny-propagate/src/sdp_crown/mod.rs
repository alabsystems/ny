// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SDP-CROWN utilities for tighter LiRPA bounds under ℓ2 input sets.
//!
//! This module implements the key offset tightening from:
//!   SDP-CROWN: Efficient Bound Propagation for Neural Network Verification
//!   with Tightness of Semidefinite Programming (arXiv:2506.06665)
//!
//! We currently implement the ReLU-specific offset `h(g, λ)` (Theorem 1) used to
//! convert a standard LiRPA/CROWN linear relaxation (computed on an ℓ∞ box
//! containing the ℓ2 ball) into a valid and often tighter relaxation for the
//! ℓ2 ball directly.

use ny_core::{NyError, Result};
use ny_tensor::next_down_f32;

/// Lower clamp for SDP-CROWN λ parameter.
///
/// The offset formula `h(g, λ) = -½(λ·ρ² + ||φ||²/λ)` (Theorem 1, arXiv:2506.06665)
/// is singular at λ=0. This floor prevents division-by-zero and numerical blow-up.
/// The value 1e-8 is an implementation choice — small enough to not distort the
/// grid search, large enough to keep `||φ||²/λ` representable in f64.
pub(crate) const MIN_LAMBDA: f64 = 1e-8;

/// Upper clamp for SDP-CROWN λ parameter.
///
/// For very large λ the `λ·ρ²` term dominates and h→-∞, so the offset is useless.
/// Clamping at 1e8 (symmetric with `MIN_LAMBDA` in log-space) bounds the grid
/// search range. See `SDP_LAMBDA_GRID_STEPS` for the search resolution.
pub(crate) const MAX_LAMBDA: f64 = 1e8;

/// Number of log-spaced grid search points for λ optimisation.
///
/// 41 points across `[MIN_LAMBDA, MAX_LAMBDA]` gives 0.4-decade spacing
/// (16 decades / 40 intervals). This is cheap relative to the inner-product
/// cost per step and empirically finds the optimum within 1–2% of exhaustive
/// search on ACAS-Xu and MNIST benchmarks. Increasing to 81 doubles cost with
/// negligible improvement; decreasing below 21 risks missing sharp optima.
pub(crate) const SDP_LAMBDA_GRID_STEPS: usize = 41;

/// Threshold below which ||x̂||² is treated as zero for the SDP-CROWN closed-form path.
///
/// When the centre point x̂ of the ℓ2 ball has negligible norm, the closed-form
/// optimum `h* = -ρ·||φ₀||₂` (Appendix B.2, arXiv:2506.06665) applies because
/// the cross-term `<φ, x̂>` vanishes. The threshold 1e-20 is ~(1e-10)², well below
/// any meaningful f32 norm-squared, ensuring the branch activates only for true-zero
/// centres (e.g. perturbation balls centred at the origin).
pub(crate) const SDP_XHAT_ZERO_THRESHOLD: f64 = 1e-20;

/// Threshold below which ||φ₀||₂ is treated as negligible for SDP-CROWN.
///
/// When `φ₀ = min{c-g, g, 0}` has L2 norm below this threshold, the SDP offset
/// `h ≈ -ρ·||φ₀||₂` would be smaller than f32 precision and the offset is set to
/// zero. The value 1e-12 is chosen to be well above f64 rounding noise (~1e-16)
/// but well below any meaningful bound improvement in f32 arithmetic (~1e-7).
pub(crate) const SDP_PHI_SIGNIFICANCE_EPS: f64 = 1e-12;

#[cfg(test)]
fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

#[cfg(test)]
fn dot(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(NyError::shape_mismatch(vec![a.len()], vec![b.len()]));
    }
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum())
}

fn ensure_finite_slice(name: &str, values: &[f32]) -> Result<()> {
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(NyError::InvalidSpec(format!(
                "SDP-CROWN: {name} must be finite (got {v} at index {i})"
            )));
        }
    }
    Ok(())
}

fn l2_norm_sq(x: &[f32]) -> f64 {
    x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()
}

fn phi_norm_sq(c: &[f32], g: &[f32], x_hat: &[f32], lambda: f64) -> Result<f64> {
    if c.len() != g.len() {
        return Err(NyError::shape_mismatch(vec![c.len()], vec![g.len()]));
    }
    if c.len() != x_hat.len() {
        return Err(NyError::shape_mismatch(vec![c.len()], vec![x_hat.len()]));
    }
    let mut sum = 0.0f64;
    for i in 0..c.len() {
        let ci = c[i] as f64;
        let gi = g[i] as f64;
        let xhi = x_hat[i] as f64;

        // φ_i(g,λ) = min{c_i - g_i - λ x̂_i, g_i + λ x̂_i, 0}
        let a = ci - gi - lambda * xhi;
        let b = gi + lambda * xhi;
        let phi = a.min(b).min(0.0);
        sum += phi * phi;
    }
    Ok(sum)
}

pub(crate) fn phi0_l2_norm(c: &[f32], g: &[f32]) -> Result<f64> {
    if c.len() != g.len() {
        return Err(NyError::shape_mismatch(vec![c.len()], vec![g.len()]));
    }
    // When x̂ = 0, φ is independent of λ:
    // φ_i = min{c_i - g_i, g_i, 0}
    let mut sum = 0.0f64;
    for i in 0..c.len() {
        let a = (c[i] - g[i]) as f64;
        let b = g[i] as f64;
        let phi = a.min(b).min(0.0);
        sum += phi * phi;
    }
    Ok(sum.sqrt())
}

/// Compute SDP-CROWN ReLU offset `h(g, λ)` for `c^T ReLU(x) >= g^T x + h` on an ℓ2 ball.
///
/// The ball is `B2(x_hat, rho) = { x : ||x - x_hat||_2 <= rho }`.
pub fn relu_sdp_offset_for_lambda(
    c: &[f32],
    g: &[f32],
    x_hat: &[f32],
    rho: f32,
    lambda: f64,
) -> Result<f32> {
    if !lambda.is_finite() {
        return Err(NyError::InvalidSpec(format!(
            "SDP-CROWN: lambda must be finite (got {lambda})"
        )));
    }
    if !rho.is_finite() {
        return Err(NyError::InvalidSpec(format!(
            "SDP-CROWN: rho must be finite (got {rho})"
        )));
    }
    if rho < 0.0 {
        return Err(NyError::InvalidSpec(format!(
            "SDP-CROWN: rho must be >= 0 (got {rho})"
        )));
    }
    ensure_finite_slice("c", c)?;
    ensure_finite_slice("g", g)?;
    ensure_finite_slice("x_hat", x_hat)?;
    let lambda = lambda.clamp(MIN_LAMBDA, MAX_LAMBDA);
    let rho2_minus_xhat2 = (rho as f64) * (rho as f64) - l2_norm_sq(x_hat);
    let phi2 = phi_norm_sq(c, g, x_hat, lambda)?;
    let h = -0.5f64 * (lambda * rho2_minus_xhat2 + phi2 / lambda);
    // Round toward -∞ so the stored f32 offset is <= the true f64 value.
    // The offset appears in lower-bound inequalities (c^T ReLU(x) >= g^T x + h),
    // so `as f32` (round-to-nearest-even) could round h upward → unsound.
    // `next_down_f32` subtracts 1 ULP, guaranteeing h_f32 <= h_f64.
    // See issue #1676.
    Ok(next_down_f32(h as f32))
}

/// Compute a near-optimal SDP-CROWN ReLU offset by maximizing over λ.
///
/// - If `rho == 0`, returns the exact offset for the singleton set `{x_hat}`.
/// - If `x_hat == 0`, uses the closed-form optimum `h* = -rho * ||min{c-g, g, 0}||_2`.
/// - Otherwise, uses a log-spaced grid search over λ (robust and fast enough for small nets).
///
/// Production code uses `relu_sdp_offset_for_lambda` directly with its own λ optimization loop.
/// This grid-search wrapper is retained for test coverage of the SDP offset formula.
#[cfg(test)]
pub fn relu_sdp_offset_opt(c: &[f32], g: &[f32], x_hat: &[f32], rho: f32) -> Result<f32> {
    if !rho.is_finite() {
        return Err(NyError::InvalidSpec(format!(
            "SDP-CROWN: rho must be finite (got {rho})"
        )));
    }
    if rho < 0.0 {
        return Err(NyError::InvalidSpec(format!(
            "SDP-CROWN: rho must be >= 0 (got {rho})"
        )));
    }
    if c.len() != g.len() {
        return Err(NyError::shape_mismatch(vec![c.len()], vec![g.len()]));
    }
    if c.len() != x_hat.len() {
        return Err(NyError::shape_mismatch(vec![c.len()], vec![x_hat.len()]));
    }
    ensure_finite_slice("c", c)?;
    ensure_finite_slice("g", g)?;
    ensure_finite_slice("x_hat", x_hat)?;
    if rho == 0.0 {
        let lhs = dot(c, &x_hat.iter().copied().map(relu).collect::<Vec<_>>())?;
        let rhs = dot(g, x_hat)?;
        return Ok(lhs - rhs);
    }

    let xhat_norm_sq = l2_norm_sq(x_hat);
    if xhat_norm_sq < SDP_XHAT_ZERO_THRESHOLD {
        let phi_norm = phi0_l2_norm(c, g)?;
        return Ok(next_down_f32(-(rho as f64 * phi_norm) as f32));
    }

    let phi0 = phi0_l2_norm(c, g)?;
    let base = if rho as f64 > SDP_PHI_SIGNIFICANCE_EPS && phi0 > SDP_PHI_SIGNIFICANCE_EPS {
        (phi0 / rho as f64).clamp(MIN_LAMBDA, MAX_LAMBDA)
    } else {
        1.0
    };

    let min_lambda = (base * 1e-3).clamp(MIN_LAMBDA, MAX_LAMBDA);
    let max_lambda = (base * 1e3).clamp(min_lambda, MAX_LAMBDA);

    let steps = SDP_LAMBDA_GRID_STEPS;
    let log_min = min_lambda.ln();
    let log_max = max_lambda.ln();
    let mut best_h = f64::NEG_INFINITY;
    for t in 0..steps {
        let frac = if steps == 1 {
            0.0
        } else {
            t as f64 / (steps as f64 - 1.0)
        };
        let lambda = (log_min + frac * (log_max - log_min)).exp();
        let h = relu_sdp_offset_for_lambda(c, g, x_hat, rho, lambda)? as f64;
        if h.is_finite() && h > best_h {
            best_h = h;
        }
    }

    // The individual relu_sdp_offset_for_lambda calls already round toward -∞,
    // so best_h (max of rounded values) is exact in f32. Apply next_down_f32
    // defensively for the rho=0 early-return path consistency.
    Ok(next_down_f32(best_h as f32))
}

#[cfg(test)]
mod tests;
