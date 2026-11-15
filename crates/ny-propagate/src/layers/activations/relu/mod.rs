// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2, ArrayD};
use ny_core::dd::next_up_f64;
use ny_core::{nan_propagating_max, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;
use tracing::debug;

use super::{relu_crossing_upper_chord, LinearRelaxation};
use crate::layers::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched, nan_only_domain_guard,
    non_finite_domain_guard, BoundPropagation,
};
use crate::layers::linear::bias::{add_f64_down, add_f64_up, f32_to_f64_exact};
use crate::{contiguous_flat_slice, BatchedLinearBounds, LinearBounds};

mod ibp;
use ibp::relu_ibp;

/// Minimum width for ReLU relaxation denominator to avoid division by zero.
/// Matches α,β-CROWN Python baseline: `/ (upper - lower + 1e-8)`.
/// See: auto_LiRPA/operators/relu.py
pub(crate) const RELU_RELAX_MIN_WIDTH: f32 = 1e-8;

/// Reachability audit for the α-linear ReLU backward incoming-err DROP (task #35).
///
/// `propagate_linear_with_alpha` silently discards any certified coefficient
/// error attached to the incoming `bounds` (false-proof direction). Before
/// designing the sign-stability carry, we must know whether ANY production
/// verdict path actually reaches this entry with NONZERO incoming err. When
/// `NY_ALPHA_ERR_AUDIT` is set, this records total entries, `has_coeff_err()`
/// entries, and entries whose err matrices contain a finite nonzero value, and
/// eprintln's a running summary (first 20 nonzero hits + every power-of-two).
mod alpha_err_audit {
    use crate::LinearBounds;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TOTAL: AtomicU64 = AtomicU64::new(0);
    static HAS_ERR_FLAG: AtomicU64 = AtomicU64::new(0);
    static NONZERO_ERR: AtomicU64 = AtomicU64::new(0);
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

    fn enabled() -> bool {
        // Read the env once; the per-call cost when disabled is a cached load.
        *ENABLED.get_or_init(|| std::env::var_os("NY_ALPHA_ERR_AUDIT").is_some())
    }

    /// Largest finite entry across the optional err matrices, and whether any
    /// non-finite (Inf/NaN) err entry was present.
    fn err_stats(b: &LinearBounds) -> (f64, bool) {
        let mut max = 0.0f64;
        let mut nonfinite = false;
        for m in [b.lower_a_err(), b.upper_a_err()].into_iter().flatten() {
            for &v in m.iter() {
                if v.is_finite() {
                    if (v as f64) > max {
                        max = v as f64;
                    }
                } else {
                    nonfinite = true;
                }
            }
        }
        (max, nonfinite)
    }

    pub(super) fn record(bounds: &LinearBounds) {
        if !enabled() {
            return;
        }
        let total = TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
        if !bounds.has_coeff_err() {
            if total.is_multiple_of(200_000) {
                eprintln!(
                    "[NY_ALPHA_ERR_AUDIT] total={} has_err_flag={} nonzero_err={}",
                    total,
                    HAS_ERR_FLAG.load(Ordering::Relaxed),
                    NONZERO_ERR.load(Ordering::Relaxed),
                );
            }
            return;
        }
        HAS_ERR_FLAG.fetch_add(1, Ordering::Relaxed);
        let (max_err, nonfinite) = err_stats(bounds);
        if max_err > 0.0 || nonfinite {
            let n = NONZERO_ERR.fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 20 || n.is_power_of_two() {
                eprintln!(
                    "[NY_ALPHA_ERR_AUDIT] NONZERO incoming err #{n}: max_err={max_err:.3e} \
                     nonfinite={nonfinite} shape={}x{} (total_entries={total})",
                    bounds.num_outputs(),
                    bounds.num_inputs(),
                );
            }
        }
    }
}

/// Compute ReLU linear relaxation for a single neuron given pre-activation bounds [l, u].
///
/// Returns (lower_slope, lower_intercept, upper_slope, upper_intercept) as a LinearRelaxation.
/// Made pub(crate) for GPU CROWN layer descriptor extraction.
#[inline]
pub(crate) fn relu_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    if l.is_nan() || u.is_nan() {
        // NaN bounds → (-inf, +inf) intercepts so CROWN drives bounds to ±inf (sound).
        return LinearRelaxation::nan_fallback();
    }

    if l >= 0.0 {
        return LinearRelaxation::identity();
    }
    if u <= 0.0 {
        return LinearRelaxation::zero();
    }
    if l.is_infinite() && u.is_infinite() {
        // No finite affine upper envelope exists on (-inf, +inf); fail closed.
        return LinearRelaxation::new(0.0, 0.0, 0.0, f32::INFINITY);
    }
    if u.is_infinite() {
        // For finite l < 0 < +inf, chord limit gives slope -> 1 and intercept -> -l.
        return LinearRelaxation::new(1.0, 0.0, 1.0, -l);
    }
    if l.is_infinite() {
        // For -inf < 0 < finite u, tight finite upper envelope is the constant y <= u.
        return LinearRelaxation::new(0.0, 0.0, 0.0, u);
    }

    let (lambda, lambda_intercept) = relu_crossing_upper_chord(l, u, Some(RELU_RELAX_MIN_WIDTH));
    let alpha = if u > -l { 1.0 } else { 0.0 };
    LinearRelaxation::new(alpha, 0.0, lambda, lambda_intercept)
}

/// Per-carrier-row spec-α lookup for the dense α backward
/// (#spec-axis-alpha, design §5.2).
///
/// `slot_of_row[j]` names the materialized-α slot serving carrier row `j`
/// (rows beyond the table, and `None` entries, read the shared vector).
/// Built once per ReLU node by the walk — the row→slot resolution must never
/// sit inside the `j × i` loop (external review: K=8 over 100×8k would pay
/// ~6.4M scans per node).
#[derive(Debug, Clone, Copy)]
pub(crate) struct SpecRowAlphas<'a> {
    pub(crate) slot_of_row: &'a [Option<usize>],
    pub(crate) slot_alphas: &'a [Array1<f32>],
}

impl SpecRowAlphas<'_> {
    /// The materialized lower-α for carrier row `j`, if row `j` is active
    /// and its slot index is in range (out-of-range ⇒ shared, fail-closed).
    fn lower_alpha_for_carrier_row(&self, carrier_row: usize) -> Option<&Array1<f32>> {
        self.slot_of_row
            .get(carrier_row)
            .copied()
            .flatten()
            .and_then(|slot| self.slot_alphas.get(slot))
    }
}

/// Flat, read-only view of ONE pre-activation endpoint, in row-major logical
/// order, borrowing whenever the endpoint is already standard layout.
///
/// Replaces the `pre_activation.flatten()` → `.lower()/.upper().clone()` →
/// `into_dimensionality::<Ix1>()` chain that both dense CROWN backward entries
/// ran on entry. That chain allocated and filled FOUR full-length buffers per
/// call — the two `iter().copied().collect()`s inside `BoundedTensor::flatten`,
/// then a `clone()` of each flattened endpoint — and built two `TrackedArrayD`
/// endpoints it discarded, to produce arrays that are only ever READ (indexed
/// in the relaxation and backward loops, then handed to
/// `fold_coeff_err_into_bias` as slices). ReLU is the most-executed layer in
/// the verifier, so that ran per node, per backward pass, per domain.
///
/// BIT-IDENTICAL, not merely close. [`contiguous_flat_slice`] returns
/// `as_slice()` when the endpoint is standard layout, and ndarray's
/// `as_slice()` is `Some` ONLY for a standard-layout array — whose memory
/// order IS the row-major logical order `iter()` walks; otherwise it falls
/// back to the SAME `iter().copied().collect()` that `flatten()` performed.
/// Either arm therefore presents element `i` as the identical `f32` at the
/// identical index, so every downstream read, product and accumulation keeps
/// its original operand and its original order. Only the data movement is
/// removed; no arithmetic is reordered, fused, or re-associated.
#[cfg(not(test))]
#[inline]
fn pre_activation_flat(endpoint: &ArrayD<f32>) -> Cow<'_, [f32]> {
    contiguous_flat_slice(endpoint)
}

/// Test build of [`pre_activation_flat`]: identical result, plus the
/// engagement counter the zero-copy tests assert on.
#[cfg(test)]
fn pre_activation_flat(endpoint: &ArrayD<f32>) -> Cow<'_, [f32]> {
    let flat = contiguous_flat_slice(endpoint);
    flat_reads::record(matches!(flat, Cow::Borrowed(_)));
    flat
}

/// Proof that a backward pass actually took the borrow, rather than only that
/// two numbers agreed: reinstating `flatten()` would still produce equal
/// floats, so an equality-only test cannot fail closed on the copies coming
/// back. Counters are THREAD-LOCAL, so a parallel test driving the same
/// backward cannot pollute another test's delta. The whole module is
/// `cfg(test)`; release builds take the counter-free definition above.
#[cfg(test)]
mod flat_reads {
    use std::cell::Cell;

    thread_local! {
        static BORROWED: Cell<usize> = const { Cell::new(0) };
        static COPIED: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn record(borrowed: bool) {
        if borrowed {
            BORROWED.with(|c| c.set(c.get() + 1));
        } else {
            COPIED.with(|c| c.set(c.get() + 1));
        }
    }

    /// `(borrowed, copied)` pre-activation endpoint reads made on THIS thread
    /// so far. Compare two snapshots to get one call's counts.
    pub(super) fn snapshot() -> (usize, usize) {
        (BORROWED.with(Cell::get), COPIED.with(Cell::get))
    }
}

/// A ReLU activation layer.
#[derive(Debug, Clone, Default)]
pub struct ReLULayer;

impl ReLULayer {
    /// Create a new ReLU layer.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// CROWN backward propagation through ReLU with pre-activation bounds.
    ///
    /// For ReLU y = ReLU(x), with pre-activation bounds \[l, u\] for x:
    /// - If l >= 0: y = x (identity), pass-through
    /// - If u <= 0: y = 0 (zero), no dependence
    /// - If l < 0 < u: use linear relaxation
    ///   - Upper: y <= λ(x - l) where λ = u/(u-l)
    ///   - Lower: y >= α*x where α ∈ \[0,1\] (default heuristic: α=1 if u > -l, else α=0)
    ///
    /// The backward propagation handles positive/negative coefficients differently:
    /// - For positive `A[j,i]` in lower bound: want y_i large, use lower relaxation (α*x)
    /// - For negative `A[j,i]` in lower bound: want y_i small, use upper relaxation (λ*x - λ*l)
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("ReLU layer CROWN backward propagation with pre-activation bounds");
        // NaN-only guard: relu_linear_relaxation has proven over-approximation branches
        // for l=-inf and/or u=+inf (see the infinite-case arms above), so infinite
        // pre-activation bounds yield a tight sound bound instead of an IBP fallback.
        // NaN still bails (cannot be bounded).
        nan_only_domain_guard("ReLU", pre_activation)?;
        crown_elementwise_backward(bounds, pre_activation, relu_linear_relaxation)
    }

    /// SDP-CROWN backward propagation through ReLU for an ℓ2 ball constraint on pre-activations.
    ///
    /// This follows the standard CROWN/LiRPA backward pass to compute `g(α)` (slopes), but
    /// replaces the per-neuron box offset with the SDP-CROWN offset `h(g,λ)` (arXiv:2506.06665),
    /// using the ℓ2 ball `||x - x_hat||_2 <= rho` for this layer's pre-activation vector.
    ///
    /// Notes:
    /// - Currently implemented for 1-D flattened pre-activations.
    /// - Uses a lightweight 1-D search to pick a near-optimal shared λ per layer.
    pub fn propagate_linear_with_bounds_sdp(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        x_hat: &Array1<f32>,
        rho: f32,
    ) -> Result<LinearBounds> {
        debug!("ReLU layer SDP-CROWN backward propagation (ℓ2 ball offset)");
        non_finite_domain_guard("ReLU-SDP", pre_activation)?;

        // Flat, READ-ONLY endpoints — see `pre_activation_flat` for why this
        // borrow is bit-identical to the four-copy chain it replaces.
        let pre_lower_flat = pre_activation_flat(pre_activation.lower());
        let pre_upper_flat = pre_activation_flat(pre_activation.upper());
        let pre_lower: &[f32] = &pre_lower_flat;
        let pre_upper: &[f32] = &pre_upper_flat;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }
        if x_hat.len() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![x_hat.len()],
            });
        }

        let x_hat_slice = contiguous_flat_slice(x_hat);
        let xhat_norm_sq = x_hat_slice
            .iter()
            .map(|&v| {
                let fv = v as f64;
                fv * fv
            })
            .sum::<f64>();

        let num_outputs = bounds.num_outputs();

        // Compute standard ReLU box relaxation parameters (α, λ, intercept).
        let mut lambda = Array1::<f32>::zeros(num_neurons);
        let mut alpha = Array1::<f32>::zeros(num_neurons);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];

            if l.is_nan() || u.is_nan() {
                lambda[i] = 0.0;
                alpha[i] = 0.0;
            } else if l >= 0.0 {
                lambda[i] = 1.0;
                alpha[i] = 1.0;
            } else if u <= 0.0 {
                lambda[i] = 0.0;
                alpha[i] = 0.0;
            } else if l.is_infinite() && u.is_infinite() {
                lambda[i] = 0.0; // maximally loose upper (intercept handled separately)
                alpha[i] = 0.0;
            } else if u.is_infinite() {
                lambda[i] = 1.0; // chord limit: slope -> 1
                alpha[i] = 1.0;
            } else if l.is_infinite() {
                lambda[i] = 0.0; // chord limit: slope -> 0, upper = constant u
                alpha[i] = 0.0;
            } else {
                // Finite crossing case: l < 0 < u, so `u - l` is an opposite-signed
                // subtraction rather than a same-sign cancellation case. We still form
                // the width in f64 so large finite crossings do not overflow `u - l` to
                // +inf before the chord is computed. We do not apply RELU_RELAX_MIN_WIDTH
                // here; a denominator-only clamp would flatten the upper line below
                // (u, u), which is unsound. The Python baseline widens the effective
                // upper endpoint before computing the chord (auto_LiRPA/operators/relu.py:586-594),
                // not the denominator alone.
                let (upper_lambda, _) = relu_crossing_upper_chord(l, u, None);
                lambda[i] = upper_lambda;
                alpha[i] = if u > -l { 1.0 } else { 0.0 };
            }
        }

        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b_f64: Vec<f64> = bounds.lower_b().iter().map(|&v| v as f64).collect();
        let mut new_upper_b_f64: Vec<f64> = bounds.upper_b().iter().map(|&v| v as f64).collect();

        // Scratch buffers to avoid per-neuron allocations inside the inner loops.
        let mut c_prime = vec![0.0f32; num_neurons];
        let mut g_prime = vec![0.0f32; num_neurons];

        for j in 0..num_outputs {
            // Build new coefficient rows (g) for lower/upper.
            for i in 0..num_neurons {
                let la = bounds.lower_a()[[j, i]];
                let ua = bounds.upper_a()[[j, i]];

                // Lower bound transform (same as standard CROWN, but without box intercept).
                // Guard: skip zero coefficients to avoid 0*inf NaN.
                // Directed rounding on slope products: next_down_f32 for lower_a,
                // next_up_f32 for upper_a. Moves bounds away from true value (sound).
                new_lower_a[[j, i]] = if la > 0.0 {
                    next_down_f32(la * alpha[i])
                } else if la < 0.0 {
                    next_down_f32(la * lambda[i])
                } else {
                    0.0
                };

                // Upper bound transform (same as standard CROWN, but without box intercept).
                new_upper_a[[j, i]] = if ua > 0.0 {
                    next_up_f32(ua * lambda[i])
                } else if ua < 0.0 {
                    next_up_f32(ua * alpha[i])
                } else {
                    0.0
                };
            }
        }

        // Select a shared lambda for this layer (Appendix B.2: one parameter per layer).
        let layer_lambda = if rho == 0.0 {
            1.0f64
        } else if xhat_norm_sq < crate::sdp_crown::SDP_XHAT_ZERO_THRESHOLD {
            let min_lambda = crate::sdp_crown::MIN_LAMBDA;
            let max_lambda = crate::sdp_crown::MAX_LAMBDA;
            let mut phi_norm_sq_sum = 0.0f64;
            let mut count = 0usize;
            let phi_eps = crate::sdp_crown::SDP_PHI_SIGNIFICANCE_EPS;
            for j in 0..num_outputs {
                let c_lower_view = bounds.lower_a().row(j);
                let g_lower_view = new_lower_a.row(j);
                let c_lower = contiguous_flat_slice(&c_lower_view);
                let g_lower = contiguous_flat_slice(&g_lower_view);
                let phi_norm = crate::sdp_crown::phi0_l2_norm(&c_lower, &g_lower)?;
                if phi_norm > phi_eps {
                    phi_norm_sq_sum += phi_norm * phi_norm;
                    count += 1;
                }

                for i in 0..num_neurons {
                    let ua = bounds.upper_a()[[j, i]];
                    c_prime[i] = -ua;
                    g_prime[i] = -new_upper_a[[j, i]];
                }
                let phi_norm = crate::sdp_crown::phi0_l2_norm(&c_prime, &g_prime)?;
                if phi_norm > phi_eps {
                    phi_norm_sq_sum += phi_norm * phi_norm;
                    count += 1;
                }
            }
            if phi_norm_sq_sum > 0.0 {
                // λ* = sqrt(sum_j ||phi_j||^2 / (rho^2 * num_constraints))
                let denom = (count as f64).sqrt() * rho as f64;
                (phi_norm_sq_sum.sqrt() / denom).clamp(min_lambda, max_lambda)
            } else {
                min_lambda
            }
        } else {
            let min_lambda = crate::sdp_crown::MIN_LAMBDA;
            let max_lambda = crate::sdp_crown::MAX_LAMBDA;
            let steps = crate::sdp_crown::SDP_LAMBDA_GRID_STEPS;
            let log_min = min_lambda.ln();
            let log_max = max_lambda.ln();
            let mut best_lambda = 1.0f64;
            let mut best_score = f64::NEG_INFINITY;

            for t in 0..steps {
                let frac = if steps == 1 {
                    0.0
                } else {
                    t as f64 / (steps as f64 - 1.0)
                };
                let candidate = (log_min + frac * (log_max - log_min)).exp();
                let mut score = 0.0f64;
                for j in 0..num_outputs {
                    let c_lower_view = bounds.lower_a().row(j);
                    let g_lower_view = new_lower_a.row(j);
                    let c_lower = contiguous_flat_slice(&c_lower_view);
                    let g_lower = contiguous_flat_slice(&g_lower_view);
                    let h = crate::sdp_crown::relu_sdp_offset_for_lambda(
                        &c_lower,
                        &g_lower,
                        &x_hat_slice,
                        rho,
                        candidate,
                    )? as f64;
                    if h.is_finite() {
                        score += h;
                    }
                    for i in 0..num_neurons {
                        let ua = bounds.upper_a()[[j, i]];
                        c_prime[i] = -ua;
                        g_prime[i] = -new_upper_a[[j, i]];
                    }
                    let h_prime = crate::sdp_crown::relu_sdp_offset_for_lambda(
                        &c_prime,
                        &g_prime,
                        &x_hat_slice,
                        rho,
                        candidate,
                    )? as f64;
                    if h_prime.is_finite() {
                        score += h_prime;
                    }
                }
                if score > best_score {
                    best_score = score;
                    best_lambda = candidate;
                }
            }

            best_lambda
        };

        for j in 0..num_outputs {
            let c_lower_view = bounds.lower_a().row(j);
            let g_lower_view = new_lower_a.row(j);
            let c_lower = contiguous_flat_slice(&c_lower_view);
            let g_lower = contiguous_flat_slice(&g_lower_view);

            let h_lower = if rho == 0.0 {
                // Accumulate in f64 to avoid cancellation when relu_sum ≈ g_dot.
                let relu_sum: f64 = c_lower
                    .iter()
                    .zip(x_hat_slice.iter())
                    .map(|(&c, &xh)| c as f64 * (nan_propagating_max(xh, 0.0) as f64))
                    .sum();
                let g_dot: f64 = g_lower
                    .iter()
                    .zip(x_hat_slice.iter())
                    .map(|(&g, &xh)| g as f64 * xh as f64)
                    .sum();
                next_down_f32((relu_sum - g_dot) as f32)
            } else if xhat_norm_sq < crate::sdp_crown::SDP_XHAT_ZERO_THRESHOLD {
                let phi_norm = crate::sdp_crown::phi0_l2_norm(&c_lower, &g_lower)?;
                if phi_norm <= crate::sdp_crown::SDP_PHI_SIGNIFICANCE_EPS {
                    0.0
                } else {
                    crate::sdp_crown::relu_sdp_offset_for_lambda(
                        &c_lower,
                        &g_lower,
                        &x_hat_slice,
                        rho,
                        layer_lambda,
                    )?
                }
            } else {
                crate::sdp_crown::relu_sdp_offset_for_lambda(
                    &c_lower,
                    &g_lower,
                    &x_hat_slice,
                    rho,
                    layer_lambda,
                )?
            };
            new_lower_b_f64[j] = bounds.lower_b()[j] as f64 + h_lower as f64;

            // SDP-CROWN offset for the UPPER inequality:
            //   c^T ReLU(x) + d <= (-g')^T x + (d - h(g',λ)) where g' corresponds to -c.
            // Here, we compute (c', g') for the lower bound on (-c)^T ReLU(x):
            //   c' = -c, g' = -new_upper_a_row  (since new_upper_a is the upper-bound coefficients).
            for i in 0..num_neurons {
                let ua = bounds.upper_a()[[j, i]];
                c_prime[i] = -ua;
                g_prime[i] = -new_upper_a[[j, i]];
            }
            let h_prime = if rho == 0.0 {
                // Accumulate in f64 to avoid cancellation when relu_sum ≈ g_dot.
                let relu_sum: f64 = c_prime
                    .iter()
                    .zip(x_hat_slice.iter())
                    .map(|(&c, &xh)| c as f64 * (nan_propagating_max(xh, 0.0) as f64))
                    .sum();
                let g_dot: f64 = g_prime
                    .iter()
                    .zip(x_hat_slice.iter())
                    .map(|(&g, &xh)| g as f64 * xh as f64)
                    .sum();
                // h_prime is subtracted from upper_b, so it acts as a lower bound
                // on the offset — use next_down_f32 so the subtraction is conservative.
                next_down_f32((relu_sum - g_dot) as f32)
            } else if xhat_norm_sq < crate::sdp_crown::SDP_XHAT_ZERO_THRESHOLD {
                let phi_norm = crate::sdp_crown::phi0_l2_norm(&c_prime, &g_prime)?;
                if phi_norm <= crate::sdp_crown::SDP_PHI_SIGNIFICANCE_EPS {
                    0.0
                } else {
                    crate::sdp_crown::relu_sdp_offset_for_lambda(
                        &c_prime,
                        &g_prime,
                        &x_hat_slice,
                        rho,
                        layer_lambda,
                    )?
                }
            } else {
                crate::sdp_crown::relu_sdp_offset_for_lambda(
                    &c_prime,
                    &g_prime,
                    &x_hat_slice,
                    rho,
                    layer_lambda,
                )?
            };
            new_upper_b_f64[j] = bounds.upper_b()[j] as f64 - h_prime as f64;
        }

        // Note: we intentionally do NOT add the standard box intercept contributions here.
        // The SDP-CROWN offset replaces those terms.

        LinearBounds::new_or_conservative(
            new_lower_a,
            Array1::from_vec(
                new_lower_b_f64
                    .iter()
                    .map(|&v| next_down_f32(v as f32))
                    .collect(),
            ),
            new_upper_a,
            Array1::from_vec(
                new_upper_b_f64
                    .iter()
                    .map(|&v| next_up_f32(v as f32))
                    .collect(),
            ),
        )
    }

    /// CROWN backward propagation with explicit α values for α-CROWN optimization.
    ///
    /// Same as `propagate_linear_with_bounds` but uses provided α values instead of heuristic.
    /// Also returns gradients ∂bounds/∂α for optimization.
    ///
    /// This path intentionally remains custom: the canonical `crown_elementwise_backward`
    /// helper does not expose per-neuron gradient accumulation for α parameters.
    ///
    /// # Dual Alpha
    ///
    /// When `alpha_upper` is `Some`, uses separate alpha values for the upper bound path
    /// (`ua < 0` case), matching the reference alpha-beta-CROWN which stores `alpha[0]`
    /// for the lower bound path and `alpha[1]` for the upper bound path. This allows
    /// independent optimization of lower relaxation slopes for each bound direction.
    ///
    /// When `alpha_upper` is `None`, uses `alpha` for both paths (legacy behavior).
    ///
    /// Reference: auto_LiRPA/operators/relu.py:647-652 (`selected_alpha[0]`/`[1]`)
    ///
    /// Returns: (new_bounds, gradient_lower, gradient_upper)
    /// - `gradient_lower[i]` = ∂(sum of lower bounds)/∂α_lower[i]
    /// - `gradient_upper[i]` = ∂(sum of upper bounds)/∂α_upper[i]
    pub fn propagate_linear_with_alpha(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        alpha: &Array1<f32>,
        alpha_upper: Option<&Array1<f32>>,
    ) -> Result<(LinearBounds, Array1<f32>, Array1<f32>)> {
        self.propagate_linear_with_alpha_impl(
            bounds,
            pre_activation,
            alpha,
            alpha_upper,
            None,
            None,
            true,
        )
    }

    /// Bound-only counterpart of [`Self::propagate_linear_with_alpha`].
    ///
    /// It executes the identical certified coefficient/bias arithmetic while
    /// omitting gradient allocation and accumulation. This is used only by an
    /// alpha loop iteration whose state cannot feed another evaluated bound.
    pub(crate) fn propagate_linear_with_alpha_bound_only(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        alpha: &Array1<f32>,
        alpha_upper: Option<&Array1<f32>>,
    ) -> Result<LinearBounds> {
        self.propagate_linear_with_alpha_impl(
            bounds,
            pre_activation,
            alpha,
            alpha_upper,
            None,
            None,
            false,
        )
        .map(|(bounds, _, _)| bounds)
    }

    /// Spec-row-aware variant (#spec-axis-alpha, design §5.2): active carrier
    /// rows read their materialized per-spec LOWER α; every other row — and
    /// the upper path unconditionally (design: upper stays shared until a
    /// spec-upper δ is deliberately implemented) — reads the shared vectors.
    /// `slot_of_row[j]` indexes into `slot_alphas`; rows beyond the table's
    /// length are shared. Both entry points funnel here so a δ optimized
    /// under the gradient pass and a bound published by the bound-only pass
    /// see identical geometry.
    ///
    /// `spec_gradients` (slice 2b): when `Some`, receives the PER-SLOT
    /// compose-local lower gradient — row `slot`'s own `∂bound/∂α` at
    /// EXPANDED width (`K × num_neurons`), i.e. exactly the contribution the
    /// shared `gradient_lower` sums away across rows. The shared gradient is
    /// still accumulated over ALL rows (base α remains jointly steered; δ is
    /// the per-row correction on top — design §3).
    pub(crate) fn propagate_linear_with_alpha_spec_rows(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        alpha: &Array1<f32>,
        alpha_upper: Option<&Array1<f32>>,
        spec_rows: SpecRowAlphas<'_>,
        spec_gradients: Option<&mut Array2<f32>>,
        track_gradients: bool,
    ) -> Result<(LinearBounds, Array1<f32>, Array1<f32>)> {
        self.propagate_linear_with_alpha_impl(
            bounds,
            pre_activation,
            alpha,
            alpha_upper,
            Some(spec_rows),
            spec_gradients,
            track_gradients,
        )
    }

    fn propagate_linear_with_alpha_impl(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        alpha: &Array1<f32>,
        alpha_upper: Option<&Array1<f32>>,
        spec_rows: Option<SpecRowAlphas<'_>>,
        mut spec_gradients: Option<&mut Array2<f32>>,
        track_gradients: bool,
    ) -> Result<(LinearBounds, Array1<f32>, Array1<f32>)> {
        debug!("ReLU layer α-CROWN backward propagation");
        // Task #35 reachability audit: does any production path reach here with
        // NONZERO incoming certified coefficient error (which this fn drops)?
        alpha_err_audit::record(bounds);
        non_finite_domain_guard("ReLU-alpha", pre_activation)?;

        // Flat, READ-ONLY endpoints — see `pre_activation_flat` for why this
        // borrow is bit-identical to the four-copy chain it replaces.
        let pre_lower_flat = pre_activation_flat(pre_activation.lower());
        let pre_upper_flat = pre_activation_flat(pre_activation.upper());
        let pre_lower: &[f32] = &pre_lower_flat;
        let pre_upper: &[f32] = &pre_upper_flat;

        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }
        if alpha.len() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![alpha.len()],
            });
        }
        if let Some(au) = alpha_upper {
            if au.len() != num_neurons {
                return Err(NyError::ShapeMismatch {
                    expected: vec![num_neurons],
                    got: vec![au.len()],
                });
            }
        }

        let num_outputs = bounds.num_outputs();

        // Compute upper bound relaxation parameters (lambda) for crossing neurons
        let mut lambda = Array1::<f32>::zeros(num_neurons);
        let mut lambda_intercept = Array1::<f32>::zeros(num_neurons);
        let mut is_crossing = Array1::<bool>::from_elem(num_neurons, false);

        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];

            if l.is_nan() || u.is_nan() {
                // NaN bounds: fail closed with maximally-loose upper intercept.
                // This preserves soundness for positive upper coefficients.
                lambda[i] = 0.0;
                lambda_intercept[i] = f32::INFINITY;
            } else if l >= 0.0 {
                // Always positive: identity
                lambda[i] = 1.0;
                lambda_intercept[i] = 0.0;
            } else if u <= 0.0 {
                // Always negative: zero
                lambda[i] = 0.0;
                lambda_intercept[i] = 0.0;
            } else if l.is_infinite() && u.is_infinite() {
                // No finite affine envelope; fail closed with maximally loose upper.
                lambda[i] = 0.0;
                lambda_intercept[i] = f32::INFINITY;
                is_crossing[i] = true;
            } else if u.is_infinite() {
                // l < 0, u = +inf: chord limit slope -> 1, intercept -> -l.
                lambda[i] = 1.0;
                lambda_intercept[i] = -l;
                is_crossing[i] = true;
            } else if l.is_infinite() {
                // l = -inf, u > 0: constant upper y <= u.
                lambda[i] = 0.0;
                lambda_intercept[i] = u;
                is_crossing[i] = true;
            } else {
                // Crossing: linear relaxation. As above, l < 0 < u is not a same-sign
                // cancellation case, but the raw f32 subtraction can still overflow for
                // very large finite endpoints. Compute the chord from an f64 width and
                // keep the unclamped denominator; a denominator-only RELU_RELAX_MIN_WIDTH
                // clamp would make the upper chord too flat and under-approximate ReLU(u).
                let (upper_lambda, upper_intercept) = relu_crossing_upper_chord(l, u, None);
                lambda[i] = upper_lambda;
                lambda_intercept[i] = upper_intercept;
                is_crossing[i] = true;
            }
        }

        // Backward propagation with provided alpha values
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        // CERTIFIED CONSTANT (#vnncomp-aw-constant). The comment that used to
        // stand here read "Match canonical helper precision: accumulate bias in
        // f64 to reduce cancellation" — a measurement claim that is true of the
        // PRECISION and false as a CERTIFICATE. Higham Thm 3.1 bounds the
        // binary64 error of a length-`d` sum by `gamma^f64_d * SUM|summand|`,
        // scaled by the MAGNITUDE of the terms, while the outward binary32 cast
        // at publication allows only `~2^-24 * |result|`. Under cancellation the
        // result collapses toward zero and that allowance collapses with it; the
        // true error does not. Worked shape, reachable here with `num_neurons =
        // 4097` (`certified_constant_tests` builds it): a seed of `+2^60 * i`
        // followed by 4096 terms of `-i` and one of `-2^60 * i` sums to EXACTLY
        // 0.0 in binary64 — `ulp(2^60 * i) = 256` swallows every unit term —
        // where the truth is `-4096 * i`. Published as a lower bias 4096 ABOVE
        // the truth: a false proof, on the most-executed backward in the
        // verifier.
        //
        // The fix is idiom (b) of this crate's constant-certificate idioms
        // (`layers/linear/bias.rs`), the SAME one the three certified
        // element-wise twins take (`layers/common/crown_dense.rs`,
        // `layers/common/crown_batched.rs`,
        // `layers/trigonometric/s_shaped/shared.rs`): every accumulation below
        // takes a DIRECTED binary64 step after each addition — `add_f64_down` on
        // the lower accumulators, `add_f64_up` on the upper and on the
        // non-negative error accumulators — so each accumulator is a certified
        // bound on the exact real partial sum at EVERY step, not merely at the
        // end. Idiom (b) rather than (a)'s magnitude-sum + gamma charge because
        // each `a * lambda_intercept` term is an f32 x f32 product formed in f64
        // and hence EXACT (24 + 24 <= 53 significand bits), so the accumulation
        // is the ONLY error source and there is nothing else for a magnitude sum
        // to cover.
        //
        // Seeds decode with `f32_to_f64_exact` rather than `as f64`: a plain
        // hardware conversion may observe a binary32-subnormal seed as signed
        // zero on a DAZ host, which for a lower bias reads `-1.4e-45` as `0.0`
        // and moves the bound INWARD.
        let mut new_lower_b_f64 = bounds.lower_b().mapv(f32_to_f64_exact);
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b_f64 = bounds.upper_b().mapv(f32_to_f64_exact);

        // Track which output rows have non-finite coefficients after coeff × slope.
        // When accumulated Inf coefficients multiply slopes, f32 can produce Inf/NaN.
        // Affected rows get zeroed A-coefficients and ±Inf bias (sound conservative fallback),
        // preserving precision in all other rows. Matches canonical pattern from
        // crown_elementwise_backward_indexed and crown_single.
        let mut lower_nonfinite_rows = vec![false; num_outputs];
        let mut upper_nonfinite_rows = vec![false; num_outputs];

        // Dual alpha: separate gradients for lower and upper bound paths.
        // gradient_lower[i] = ∂(sum of lower bounds)/∂α_lower[i]  (from la > 0 case)
        // gradient_upper[i] = ∂(sum of upper bounds)/∂α_upper[i]  (from ua < 0 case)
        // Reference: auto_LiRPA/operators/relu.py:647-652
        let gradient_len = if track_gradients { num_neurons } else { 0 };
        let mut gradient_lower = Array1::<f32>::zeros(gradient_len);
        let mut gradient_upper = Array1::<f32>::zeros(gradient_len);

        // Sign-stability certified coefficient-error carry (task #35,
        // #cgan-coeff-err-fold). The incoming `bounds` may carry a certified
        // per-coefficient error attached by a sound Conv/ConvTranspose/activation
        // backward. Dropping it is the FALSE-PROOF direction (a downstream layer
        // then reads the composed f32 coefficient as exact and under-counts the
        // true distance to the real coefficient). The naive faithful carry
        // (`err·(|slope_l|+|slope_u|)` for EVERY neuron, matching
        // `crown_elementwise_backward`) doubles the carried err per ReLU even for
        // stable-identity neurons → 2^L growth → regressed metaroom spec_114.
        //
        // Refinement: the composed backward selects the lower/upper envelope by
        // the SIGN of the incoming coefficient `a`. When `|a| > err_a` the true
        // coefficient `a_true ∈ [a-err_a, a+err_a]` keeps `a`'s sign, so the
        // envelope selection cannot flip and the composed error is exactly
        // `err_a·|slope_used| + gap` (no slope-sum cover, no intercept fold when
        // the selected envelope has zero intercept). Only genuinely sign-
        // ambiguous coefficients (`|a| <= err_a`) pay the hull cover
        // `err_a·(|slope_l|+|slope_u|)` plus the `err_a·|intercept|` bias fold.
        // `gap = |a_f64·slope_f64 − stored_f32|` is the directed-rounding distance
        // of the composed f32 product (present per-entry, matches crown_dense).
        //
        // When the incoming bounds carry NO error, this whole block is skipped
        // and no err is attached. It used to say the result was then
        // "byte-identical to the pre-#35 code"; #vnncomp-aw-constant falsified
        // that half of the claim — the CONSTANT accumulation below now takes a
        // directed binary64 step per addition, so a row with cancelling
        // intercept contributions publishes a strictly WIDER bias than the
        // pre-#35 code did, err or no err. The skip is still exact: it changes
        // nothing about the err planes.
        let track_err = bounds.has_coeff_err();
        let in_lower_err = bounds.lower_a_err();
        let in_upper_err = bounds.upper_a_err();
        let (mut new_lower_a_err, mut new_upper_a_err) = if track_err {
            (
                Array2::<f32>::zeros((num_outputs, num_neurons)),
                Array2::<f32>::zeros((num_outputs, num_neurons)),
            )
        } else {
            (Array2::<f32>::zeros((0, 0)), Array2::<f32>::zeros((0, 0)))
        };
        // Per-row certified intercept (bias) error: subtracted from lower_b,
        // added to upper_b (outward), accumulated in f64. These are NON-NEGATIVE
        // CHARGE accumulators, so BOTH are summed with `add_f64_up`
        // (#vnncomp-aw-constant): a round-to-nearest `+=` here UNDER-counts the
        // charge — the same Higham/ULP mismatch as the bias sums, in the
        // direction that SHRINKS a widening — and an under-counted widening is a
        // false proof exactly like an over-tight bias.
        let mut lower_b_err = vec![0.0f64; num_outputs];
        let mut upper_b_err = vec![0.0f64; num_outputs];

        // Certified composed-coefficient error for one entry. `slope_used` is the
        // envelope slope selected by the STORED coefficient's sign; `other_slope`
        // is the opposite envelope's slope. Sign-stable (`stable`) → carry only
        // `e·|slope_used| + gap`. Sign-ambiguous → hull cover
        // `e·(|slope_used|+|other_slope|) + gap`. Rounds UP to a sound f32; a
        // non-finite result (e.g. `e·∞`) becomes `+∞`, degrading the row at
        // concretize (sound).
        let coeff_err_val = |e: f32, stable: bool, slope_used: f32, other_slope: f32, gap: f64| {
            let e64 = e as f64;
            let val = if stable {
                e64 * slope_used.abs() as f64 + gap
            } else {
                // Two-line cover summed OUTWARD (see `cover_sum_up`), exactly as
                // the certified element-wise twins sum theirs. The bare
                // `|s_used| + |s_other|` is a round-to-nearest binary64 add and
                // can land BELOW the true sum whenever the two slopes differ in
                // exponent; this cover is what an ambiguous-sign coefficient's
                // whole error rests on.
                e64 * cover_sum_up(slope_used, other_slope) + gap
            };
            next_up_f32(val as f32)
        };

        for j in 0..num_outputs {
            // #spec-axis-alpha: an active carrier row reads its materialized
            // per-spec LOWER α; every other row binds the SAME shared slice
            // the non-spec path uses (bit-identical fallback — same reads,
            // same arithmetic, same order). Resolved once per row, outside
            // the neuron loop, so the inactive-row cost is one bounds check.
            let row_alpha: &Array1<f32> = spec_rows
                .as_ref()
                .and_then(|table| table.lower_alpha_for_carrier_row(j))
                .unwrap_or(alpha);
            // Slot id for this row's per-slot gradient (slice 2b), resolved
            // once per row like the α itself — never inside the neuron loop.
            let row_slot: Option<usize> = spec_rows
                .as_ref()
                .and_then(|table| table.slot_of_row.get(j).copied().flatten());
            for i in 0..num_neurons {
                let la = bounds.lower_a()[[j, i]];
                let ua = bounds.upper_a()[[j, i]];
                let l = pre_lower[i];
                let u = pre_upper[i];

                // Dual alpha: separate alpha values for lower and upper bound paths.
                // alpha_lower_i: used when la > 0 (maximizes lower bound)
                // alpha_upper_i: used when ua < 0 (minimizes upper bound)
                // Reference: auto_LiRPA/operators/relu.py selected_alpha[0] vs [1]
                // Upper path stays SHARED under #spec-axis-alpha (design:
                // lower δ must never leak into `ua < 0` behavior).
                let (alpha_lower_i, alpha_upper_i) = if l >= 0.0 {
                    (1.0, 1.0) // Always active
                } else if u <= 0.0 {
                    (0.0, 0.0) // Always inactive
                } else {
                    // Crossing: use provided alphas.
                    // The single-alpha fallback for `au` reads the SHARED
                    // vector, not `row_alpha`: lower δ must not leak into the
                    // upper path through the map_or default (#spec-axis-alpha
                    // upper-isolation; identical bytes when no spec rows are
                    // active, since row_alpha IS alpha then).
                    let al = row_alpha[i];
                    let au = alpha_upper.map_or(alpha[i], |a| a[i]);
                    (al, au)
                };

                // For lower bound output: maximize lower
                // Guard: skip zero coefficients to avoid 0*inf NaN.
                // Directed rounding on slope products: next_down_f32 for lower_a,
                // next_up_f32 for upper_a. Moves bounds away from true value (sound).
                if la > 0.0 {
                    let product = la * alpha_lower_i;
                    if product.is_finite() {
                        let stored = next_down_f32(product);
                        new_lower_a[[j, i]] = stored;
                        if track_err {
                            let e = in_lower_err.map_or(0.0, |m| m[[j, i]]);
                            if e != 0.0 {
                                // la > 0: lower envelope (slope α, intercept 0).
                                let gap =
                                    ((la as f64) * (alpha_lower_i as f64) - stored as f64).abs();
                                let stable = (la as f64) > (e as f64);
                                new_lower_a_err[[j, i]] =
                                    coeff_err_val(e, stable, alpha_lower_i, lambda[i], gap);
                                if !stable {
                                    // a_true could flip to the λ (upper) envelope,
                                    // whose intercept lowers the bound.
                                    lower_b_err[j] = add_f64_up(
                                        lower_b_err[j],
                                        e as f64 * lambda_intercept[i].abs() as f64,
                                    );
                                }
                            }
                        }
                    } else {
                        lower_nonfinite_rows[j] = true;
                    }
                    if track_gradients && is_crossing[i] {
                        // d(la * alpha * x)/d(alpha) at x = l_i gives la * l_i.
                        // The l_i < 0 factor is essential for correct sign.
                        // Reference: backward.rs AnalyticChain gradient.
                        gradient_lower[i] += la * pre_lower[i];
                        // #spec-axis-alpha slice 2b: the SAME contribution,
                        // un-summed — row `slot`'s own ∂bound/∂α, feeding its
                        // δ row instead of vanishing into the shared sum.
                        if let (Some(spec_grads), Some(slot)) = (spec_gradients.as_mut(), row_slot)
                        {
                            spec_grads[[slot, i]] += la * pre_lower[i];
                        }
                    }
                } else if la < 0.0 {
                    let product = la * lambda[i];
                    if product.is_finite() {
                        let stored = next_down_f32(product);
                        new_lower_a[[j, i]] = stored;
                        if track_err {
                            let e = in_lower_err.map_or(0.0, |m| m[[j, i]]);
                            if e != 0.0 {
                                // la < 0: upper envelope (slope λ, intercept λ_int).
                                let gap = ((la as f64) * (lambda[i] as f64) - stored as f64).abs();
                                let stable = (-(la as f64)) > (e as f64);
                                new_lower_a_err[[j, i]] =
                                    coeff_err_val(e, stable, lambda[i], alpha_lower_i, gap);
                                // The selected λ envelope's intercept always
                                // contributes; e·|λ_int| also covers the unstable
                                // flip to the 0-intercept (lower) envelope.
                                lower_b_err[j] = add_f64_up(
                                    lower_b_err[j],
                                    e as f64 * lambda_intercept[i].abs() as f64,
                                );
                            }
                        }
                    } else {
                        lower_nonfinite_rows[j] = true;
                    }
                    // Certified accumulation (#vnncomp-aw-constant); see the
                    // rationale above the accumulator declarations.
                    new_lower_b_f64[j] =
                        add_f64_down(new_lower_b_f64[j], la as f64 * lambda_intercept[i] as f64);
                } else if track_err && la == 0.0 {
                    // Stored coefficient is exactly 0, but a_true ∈ [-e, e] may
                    // select either envelope: carry the full hull cover (no gap).
                    let e = in_lower_err.map_or(0.0, |m| m[[j, i]]);
                    if e != 0.0 {
                        new_lower_a_err[[j, i]] =
                            coeff_err_val(e, false, alpha_lower_i, lambda[i], 0.0);
                        lower_b_err[j] =
                            add_f64_up(lower_b_err[j], e as f64 * lambda_intercept[i].abs() as f64);
                    }
                }

                // For upper bound output: minimize upper
                if ua > 0.0 {
                    let product = ua * lambda[i];
                    if product.is_finite() {
                        let stored = next_up_f32(product);
                        new_upper_a[[j, i]] = stored;
                        if track_err {
                            let e = in_upper_err.map_or(0.0, |m| m[[j, i]]);
                            if e != 0.0 {
                                // ua > 0: upper envelope (slope λ, intercept λ_int).
                                let gap = ((ua as f64) * (lambda[i] as f64) - stored as f64).abs();
                                let stable = (ua as f64) > (e as f64);
                                new_upper_a_err[[j, i]] =
                                    coeff_err_val(e, stable, lambda[i], alpha_upper_i, gap);
                                // Selected λ envelope's intercept contributes;
                                // covers the unstable flip to the 0-intercept side.
                                upper_b_err[j] = add_f64_up(
                                    upper_b_err[j],
                                    e as f64 * lambda_intercept[i].abs() as f64,
                                );
                            }
                        }
                    } else {
                        upper_nonfinite_rows[j] = true;
                    }
                    new_upper_b_f64[j] =
                        add_f64_up(new_upper_b_f64[j], ua as f64 * lambda_intercept[i] as f64);
                } else if ua < 0.0 {
                    // Dual alpha: use alpha_upper for the ua < 0 case.
                    // Reference: auto_LiRPA/operators/clampmult.py:37-43
                    let product = ua * alpha_upper_i;
                    if product.is_finite() {
                        let stored = next_up_f32(product);
                        new_upper_a[[j, i]] = stored;
                        if track_err {
                            let e = in_upper_err.map_or(0.0, |m| m[[j, i]]);
                            if e != 0.0 {
                                // ua < 0: lower envelope (slope α_upper, intercept 0).
                                let gap =
                                    ((ua as f64) * (alpha_upper_i as f64) - stored as f64).abs();
                                let stable = (-(ua as f64)) > (e as f64);
                                new_upper_a_err[[j, i]] =
                                    coeff_err_val(e, stable, alpha_upper_i, lambda[i], gap);
                                if !stable {
                                    // a_true could flip to the λ (upper) envelope,
                                    // whose intercept raises the bound.
                                    upper_b_err[j] = add_f64_up(
                                        upper_b_err[j],
                                        e as f64 * lambda_intercept[i].abs() as f64,
                                    );
                                }
                            }
                        }
                    } else {
                        upper_nonfinite_rows[j] = true;
                    }
                    if track_gradients && is_crossing[i] {
                        // Gradient for alpha_upper: d(ua * alpha_upper * x)/d(alpha_upper)
                        // at x = l_i gives ua * l_i.
                        // ua < 0 and l < 0, so gradient_upper[i] > 0 — increasing
                        // alpha_upper increases the upper bound (makes it less tight).
                        // The optimizer negates this to minimize upper bound.
                        gradient_upper[i] += ua * pre_lower[i];
                    }
                } else if track_err && ua == 0.0 {
                    let e = in_upper_err.map_or(0.0, |m| m[[j, i]]);
                    if e != 0.0 {
                        new_upper_a_err[[j, i]] =
                            coeff_err_val(e, false, lambda[i], alpha_upper_i, 0.0);
                        upper_b_err[j] =
                            add_f64_up(upper_b_err[j], e as f64 * lambda_intercept[i].abs() as f64);
                    }
                }
            }
        }

        // Fold the certified intercept (bias) error OUTWARD before the directed
        // cast: lower decreases, upper increases. No-op when !track_err (the
        // accumulators stay 0), preserving byte-identical output.
        if track_err {
            for j in 0..num_outputs {
                // The discharge is a binary64 addition like any other — and the
                // one most likely to CANCEL, since `-lower_b_err[j]` is
                // subtracted from a bias of arbitrary sign and magnitude — so it
                // takes the same directed step (#vnncomp-aw-constant).
                new_lower_b_f64[j] = add_f64_down(new_lower_b_f64[j], -lower_b_err[j]);
                new_upper_b_f64[j] = add_f64_up(new_upper_b_f64[j], upper_b_err[j]);
            }
        }

        // Zero affected rows and set bias to ±Inf for rows with non-finite
        // coefficient overflow. Only affected rows are degraded — all other rows
        // preserve full precision. Pattern matches common/crown_dense.rs.
        let lower_affected = lower_nonfinite_rows.iter().filter(|&&r| r).count();
        let upper_affected = upper_nonfinite_rows.iter().filter(|&&r| r).count();
        if lower_affected > 0 || upper_affected > 0 {
            debug!(
                "Alpha-CROWN ReLU backward: non-finite coeff×slope overflow in {}/{} lower rows, \
                 {}/{} upper rows — falling back to ±Inf bias for affected rows",
                lower_affected, num_outputs, upper_affected, num_outputs
            );
        }
        // Outward-round the f64 bias into f32 — but preserve EXACT ZERO. An
        // unconditional `next_down_f32(0.0)` manufactures a -1e-45 bias on a
        // row whose true bias is exactly 0.0, which (a) is a needless 1-ulp
        // loosening and (b) breaks the stacked-seed invariant that zero rows
        // stay exactly zero through every admitted step (#cgan-stacked-backward
        // injection guard: a nonzero find refuses the whole stacked pass).
        //
        // CORRECTED RATIONALE (#vnncomp-aw-constant). This comment used to end
        // "`v == 0.0` casts exactly, so no widening is required there." That
        // claim is true of the CAST and was FALSE of the ACCUMULATION that
        // produced `v`, and `v == 0.0` is precisely the CANCELLING case: the one
        // where the binary32 allowance `~2^-24 * |v|` collapses to 1.4e-45 while
        // the true binary64 accumulation error does not collapse at all. Under
        // the previous round-to-nearest `+=` a row whose exact bias was
        // `-4096 * i` could reach exactly 0.0 (see the accumulation comment
        // above) and this branch then published a lower bias 4096 ABOVE truth.
        //
        // The branch is sound now only because every accumulator above takes a
        // directed binary64 step per addition. `add_f64_down`/`add_f64_up`
        // short-circuit `term == 0.0` exactly, and otherwise step strictly
        // outward, so a LOWER accumulator can hold exactly `+0.0` in only two
        // ways: every term it saw was exactly zero (the structural zero row), or
        // it stepped down from `+5e-324`, whose true sum is therefore >= 0.0.
        // Both make `0.0` a sound lower bound. Symmetrically for the upper
        // accumulator and `-0.0`. A row whose NONZERO terms merely cancel no
        // longer lands on zero at all: it ends strictly outside and is published
        // one binary32 ulp further out.
        let mut new_lower_b = new_lower_b_f64.mapv(|v| {
            if v == 0.0 {
                0.0
            } else {
                next_down_f32(v as f32)
            }
        });
        let mut new_upper_b =
            new_upper_b_f64.mapv(|v| if v == 0.0 { 0.0 } else { next_up_f32(v as f32) });
        for j in 0..num_outputs {
            if lower_nonfinite_rows[j] {
                for i in 0..num_neurons {
                    new_lower_a[[j, i]] = 0.0;
                    if track_err {
                        new_lower_a_err[[j, i]] = 0.0;
                    }
                }
                new_lower_b[j] = f32::NEG_INFINITY;
            }
            if upper_nonfinite_rows[j] {
                for i in 0..num_neurons {
                    new_upper_a[[j, i]] = 0.0;
                    if track_err {
                        new_upper_a_err[[j, i]] = 0.0;
                    }
                }
                new_upper_b[j] = f32::INFINITY;
            }
        }

        // CROWN backward NaN firewall: final safety net catches any
        // missed corruption beyond per-row tracking. When the incoming bounds
        // carry certified coefficient error, propagate the sign-stability composed
        // error (task #35); otherwise emit err-free bounds — same CONSTRUCTOR as
        // the pre-#35 path, but no longer the same bits: the certified constant
        // accumulation (#vnncomp-aw-constant) widens a cancelling row's bias.
        let out = if track_err {
            let mut b = LinearBounds::new_or_conservative_with_err(
                new_lower_a,
                new_lower_b,
                new_upper_a,
                new_upper_b,
                new_lower_a_err,
                new_upper_a_err,
            )?;
            // LOCAL DISCHARGE (task #35 throughput). Fold the sign-stability
            // composed coefficient error into the BIAS over THIS layer's
            // pre-activation box, clearing the error so the OUTPUT carries none.
            // Sound: for z ∈ [l, u] the uncertainty |Σ_i δ_i·z_i| is bounded by
            // Σ_i err_i·max(|l_i|, |u_i|) — exactly the penalty
            // `fold_coeff_err_into_bias` applies (the precise #vnncomp-aw-soundness
            // discharge), and z_i = ReLU(input) always lies in that box. This is
            // the SAME scalar penalty `concretize_sound` would later apply to the
            // carried error, so bound quality is unchanged, BUT carrying the error
            // coefficient-wise instead makes every downstream conv/linear backward
            // run the (expensive) sound err-carrier on every BaB rebound —
            // measured as an ~18x throughput regression on metaroom spec_114
            // (base unsat 17s vs carried timeout 306s, identical concurrent load).
            // Discharging here keeps soundness while restoring downstream cost.
            // The fold is unconditional. It already was: the removed
            // `flatten()` handed this site a `Vec`-backed `Array1`, whose
            // `as_slice()` is always `Some`, so the guard this replaces could
            // never fail. `pre_activation_flat` likewise always yields a flat
            // `&[f32]` (borrowed, or one contiguous copy for a non-standard
            // layout), so the same fold runs on the same bytes.
            b.fold_coeff_err_into_bias(pre_lower, pre_upper);
            b
        } else {
            LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)?
        };
        Ok((out, gradient_lower, gradient_upper))
    }

    /// Batched CROWN backward propagation through ReLU with pre-activation bounds.
    ///
    /// Same as `propagate_linear_with_bounds` but operates on N-D batched bounds,
    /// preserving batch structure [...batch, dim].
    ///
    /// For ReLU y = ReLU(x), with pre-activation bounds [l, u] for x:
    /// - If l >= 0: y = x (identity), pass-through
    /// - If u <= 0: y = 0 (zero), no dependence
    /// - If l < 0 < u: use linear relaxation
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        debug!("ReLU layer batched CROWN backward propagation");
        // NaN-only guard (see propagate_linear_with_bounds): the batched path uses the
        // same proven relu_linear_relaxation infinite-case branches.
        nan_only_domain_guard("ReLU", pre_activation)?;
        crown_elementwise_backward_batched(bounds, pre_activation, relu_linear_relaxation)
    }

    /// DOMAIN-batched CROWN backward through ReLU (#lsnc-relu STEP 2).
    ///
    /// Vectorizes the per-domain ReLU backward loop of the input-split batched
    /// engine (`backward_core::dispatch_node_backward`, the ReLU arm) into a single
    /// batched pass. Each domain `d` carries a DIFFERENT pre-activation box (so the
    /// triangle relaxation slope is box-dependent — the reason ReLU is excluded from
    /// the hull-based conv/BN domain-stack whitelist), hence this carries a
    /// per-domain `[num_neurons]` slope/intercept relaxation and applies it to that
    /// domain's `[num_outputs, num_neurons]` coefficient block. There is NO
    /// cross-neuron contraction (ReLU backward is diagonal: `new_a[j,i] = a[j,i] *
    /// slope_i`, selected by `sign(a[j,i])`), so — unlike the Linear GEMM — there is
    /// no accumulation-order hazard; the only reduction is the per-row f64 bias fold
    /// `Σ_i a[j,i]·intercept_i`, which this preserves in the SAME `i = 0..num_neurons`
    /// order as the scalar path.
    ///
    /// # Soundness — BIT-IDENTICAL
    ///
    /// For every domain returned in the `Some(..)` result, the output `LinearBounds`
    /// is BIT-IDENTICAL to running, per domain:
    ///   - `propagate_linear_with_alpha(bounds, pre, alpha_lower, Some(alpha_upper)).0`
    ///     when `alpha_batch[d]` is `Some` (gradients discarded, matching the ReLU
    ///     dispatch arm which drops them), or
    ///   - `propagate_linear_with_bounds(bounds, pre)`
    ///     when `alpha_batch[d]` is `None`.
    ///
    /// Same relaxation (`relu_crossing_upper_chord` / `relu_linear_relaxation`), same
    /// directed rounding (`next_down_f32`/`next_up_f32`), same per-row bias
    /// accumulation order AND the same per-step DIRECTED binary64 accumulation
    /// (`add_f64_down`/`add_f64_up`, #vnncomp-aw-constant), the same outward
    /// `cover_sum_up` two-line covers, the same certified coefficient-error carry,
    /// the same exact-zero-preserving publication, the same non-finite row degrade,
    /// and the same constructors — so the certified bound is neither tighter nor
    /// looser, it is the SAME f32 bits. The bit-identity is asserted by the kernel
    /// parity test in `tests_soundness.rs`; note that the HEURISTIC arm's reference
    /// is `crown_elementwise_backward` in `layers/common/crown_dense.rs`, so this
    /// file's arm must track THAT file's certificate, not merely its own twin.
    ///
    /// Returns `Ok(None)` (DECLINE → caller runs the byte-identical per-domain loop)
    /// whenever any domain is not in the clean fast-path class: non-contiguous or
    /// non-finite (NaN/±Inf) pre-activation, or a shape mismatch. Declining keeps
    /// the batched path strictly a performance transform — it can never change a
    /// bound, only defer to the reference loop (which then reproduces the exact
    /// error/Inf handling of the scalar path).
    ///
    /// `alpha_batch[d]` MUST be `Some((alpha_lower, alpha_upper))` with both arrays
    /// of length `num_neurons` for the alpha path; the caller builds them with
    /// `GraphDomainAlphaState::build_alpha_array` / `build_alpha_upper_array`, the
    /// identical source the scalar arm uses.
    #[allow(clippy::type_complexity)]
    pub(crate) fn propagate_linear_multi_domain_relu(
        &self,
        bounds_batch: &[&LinearBounds],
        pre_activations: &[&BoundedTensor],
        alpha_batch: &[Option<(Array1<f32>, Array1<f32>)>],
    ) -> Result<Option<Vec<LinearBounds>>> {
        let n_domains = bounds_batch.len();
        if n_domains == 0 {
            return Ok(Some(Vec::new()));
        }
        if pre_activations.len() != n_domains || alpha_batch.len() != n_domains {
            // Structural mismatch: decline, let the scalar arm handle it.
            return Ok(None);
        }

        let num_neurons = bounds_batch[0].num_inputs();
        let num_outputs = bounds_batch[0].num_outputs();

        // Flatten every domain's pre-activation up front; DECLINE the whole batch if
        // any is non-contiguous, shape-mismatched, or non-finite (NaN/±Inf). The
        // scalar fallback then reproduces the exact guard/Inf handling of the two
        // reference functions (`non_finite_domain_guard` for the alpha path erroring
        // on non-finite; `nan_only_domain_guard` + the proven `relu_linear_relaxation`
        // infinite branches for the heuristic path).
        let mut pre_flats: Vec<BoundedTensor> = Vec::with_capacity(n_domains);
        for (d, pre) in pre_activations.iter().enumerate() {
            let flat = pre.flatten();
            if flat.len() != num_neurons {
                return Ok(None);
            }
            if bounds_batch[d].num_inputs() != num_neurons
                || bounds_batch[d].num_outputs() != num_outputs
            {
                return Ok(None);
            }
            if let Some((al, au)) = &alpha_batch[d] {
                if al.len() != num_neurons || au.len() != num_neurons {
                    return Ok(None);
                }
            }
            let (Some(lo), Some(up)) = (flat.lower().as_slice(), flat.upper().as_slice()) else {
                return Ok(None);
            };
            if lo.iter().chain(up.iter()).any(|v| !v.is_finite()) {
                return Ok(None);
            }
            pre_flats.push(flat);
        }

        // Reused per-neuron relaxation scratch (rebuilt per domain — the box, hence
        // the slope, differs per domain). `lambda`/`lambda_int` are the UPPER-chord
        // relaxation; `alpha_lo`/`alpha_up` the lower/upper-direction lower-envelope
        // slopes (dual-α). No allocation inside the domain loop.
        let mut lambda = vec![0.0f32; num_neurons];
        let mut lambda_int = vec![0.0f32; num_neurons];
        let mut alpha_lo = vec![0.0f32; num_neurons];
        let mut alpha_up = vec![0.0f32; num_neurons];
        // Reused per-output-row scratch.
        let mut lower_b_err = vec![0.0f64; num_outputs];
        let mut upper_b_err = vec![0.0f64; num_outputs];
        let mut lower_nf = vec![false; num_outputs];
        let mut upper_nf = vec![false; num_outputs];

        let mut results: Vec<LinearBounds> = Vec::with_capacity(n_domains);
        for d in 0..n_domains {
            let flat = &pre_flats[d];
            let pre_lower = flat.lower().as_slice().expect("checked contiguous above");
            let pre_upper = flat.upper().as_slice().expect("checked contiguous above");
            let bounds = bounds_batch[d];

            let out = match &alpha_batch[d] {
                Some((al, au)) => relu_batched_alpha_domain(
                    bounds,
                    pre_lower,
                    pre_upper,
                    al,
                    au,
                    num_outputs,
                    num_neurons,
                    &mut lambda,
                    &mut lambda_int,
                    &mut alpha_lo,
                    &mut alpha_up,
                    &mut lower_b_err,
                    &mut upper_b_err,
                    &mut lower_nf,
                    &mut upper_nf,
                )?,
                None => relu_batched_heuristic_domain(
                    bounds,
                    pre_lower,
                    pre_upper,
                    num_outputs,
                    num_neurons,
                    &mut lower_b_err,
                    &mut upper_b_err,
                    &mut lower_nf,
                    &mut upper_nf,
                )?,
            };
            results.push(out);
        }
        Ok(Some(results))
    }
}

/// Outward `|a| + |b|` for the sign-flip two-line cover (#vnncomp-aw-constant).
///
/// PRIVATE COPY of `layers/common/crown_dense::cover_sum_up` (and of
/// `crown_batched`'s and `trigonometric/s_shaped/shared`'s own copies), kept in
/// step with them deliberately: the ReLU arms below are the SAME composition
/// those helpers perform, reached through a different entry point, and
/// `beta_crown::engine::graph::propagation::batched::tests_soundness` asserts
/// their published error planes match at ZERO tolerance. A cover computed
/// differently here would charge the same fold differently depending only on
/// which arm the caller took — the exact asymmetry this class of defect is made
/// of. If one copy changes, change all four.
///
/// The bare form `(a.abs() + b.abs()) as f64` performs the addition in BINARY32
/// under round-to-nearest, so it can land BELOW the true sum; `a.abs() as f64 +
/// b.abs() as f64` performs it in binary64 and can still land below it. That
/// cover multiplies the incoming coefficient error into both published halves —
/// `new_*_a_err` and, through `int_cover`, the bias widening `*_b_err` — so a
/// cover rounded the wrong way under-counts BOTH, in the false-proof direction.
/// Converting exactly first also keeps a binary32-subnormal operand alive on a
/// DAZ host, where a hardware `as f64` may read it as zero, and removes a
/// spurious binary32 overflow to `+inf` for two near-`f32::MAX` intercepts.
///
/// `sum == 0.0` returns an exact `0.0` rather than `next_up_f64(0.0) = 5e-324`:
/// a relaxation with two zero intercepts contributes exactly nothing, and
/// stepping it off zero would smear a subnormal charge across every cell.
///
/// Non-finite operands propagate: `+inf` stays `+inf`, NaN stays NaN and is
/// sanitized to `+inf` by `new_or_conservative_with_err` — both vacuous, sound.
#[inline]
fn cover_sum_up(a: f32, b: f32) -> f64 {
    let sum = f32_to_f64_exact(a).abs() + f32_to_f64_exact(b).abs();
    if sum == 0.0 {
        0.0
    } else {
        next_up_f64(sum)
    }
}

/// Certified composed-coefficient error for one α-path entry — a verbatim copy of
/// the closure in [`ReLULayer::propagate_linear_with_alpha`] so the batched path
/// carries the SAME sign-stability error bits. `slope_used` is the envelope slope
/// selected by the STORED coefficient's sign; `other_slope` the opposite envelope's.
#[inline]
fn relu_alpha_coeff_err_val(
    e: f32,
    stable: bool,
    slope_used: f32,
    other_slope: f32,
    gap: f64,
) -> f32 {
    let e64 = e as f64;
    let val = if stable {
        e64 * slope_used.abs() as f64 + gap
    } else {
        // Outward two-line cover — byte-for-byte the closure's
        // (#vnncomp-aw-constant). An arm that covers differently from its twin
        // is the same asymmetry one level down.
        e64 * cover_sum_up(slope_used, other_slope) + gap
    };
    next_up_f32(val as f32)
}

/// One domain of the α-CROWN ReLU backward — a faithful transcription of
/// [`ReLULayer::propagate_linear_with_alpha`] (finite pre-activation only; the
/// gradient outputs are NOT computed since the dispatch arm discards them).
/// Writes into a fresh per-domain result; the passed-in scratch is cleared and
/// reused across domains.
#[allow(clippy::too_many_arguments)]
fn relu_batched_alpha_domain(
    bounds: &LinearBounds,
    pre_lower: &[f32],
    pre_upper: &[f32],
    alpha: &Array1<f32>,
    alpha_upper: &Array1<f32>,
    num_outputs: usize,
    num_neurons: usize,
    lambda: &mut [f32],
    lambda_int: &mut [f32],
    alpha_lo: &mut [f32],
    alpha_up: &mut [f32],
    lower_b_err: &mut [f64],
    upper_b_err: &mut [f64],
    lower_nf: &mut [bool],
    upper_nf: &mut [bool],
) -> Result<LinearBounds> {
    // Task #35 reachability audit parity with the scalar path.
    alpha_err_audit::record(bounds);

    // Per-neuron relaxation (upper chord λ + dual-α lower slopes). Finite l,u only
    // (non-finite declined upstream), so only the identity/zero/crossing arms fire.
    for i in 0..num_neurons {
        let l = pre_lower[i];
        let u = pre_upper[i];
        if l >= 0.0 {
            lambda[i] = 1.0;
            lambda_int[i] = 0.0;
            alpha_lo[i] = 1.0;
            alpha_up[i] = 1.0;
        } else if u <= 0.0 {
            lambda[i] = 0.0;
            lambda_int[i] = 0.0;
            alpha_lo[i] = 0.0;
            alpha_up[i] = 0.0;
        } else {
            let (upper_lambda, upper_intercept) = relu_crossing_upper_chord(l, u, None);
            lambda[i] = upper_lambda;
            lambda_int[i] = upper_intercept;
            alpha_lo[i] = alpha[i];
            alpha_up[i] = alpha_upper[i];
        }
    }

    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    // Certified constant + exact seed decode, byte-for-byte the scalar arm's
    // (#vnncomp-aw-constant); see the rationale in
    // `propagate_linear_with_alpha_impl`, which this function transcribes.
    let mut new_lower_b_f64 = bounds.lower_b().mapv(f32_to_f64_exact);
    let mut new_upper_b_f64 = bounds.upper_b().mapv(f32_to_f64_exact);

    let track_err = bounds.has_coeff_err();
    let in_lower_err = bounds.lower_a_err();
    let in_upper_err = bounds.upper_a_err();
    let (mut new_lower_a_err, mut new_upper_a_err) = if track_err {
        (
            Array2::<f32>::zeros((num_outputs, num_neurons)),
            Array2::<f32>::zeros((num_outputs, num_neurons)),
        )
    } else {
        (Array2::<f32>::zeros((0, 0)), Array2::<f32>::zeros((0, 0)))
    };
    for j in 0..num_outputs {
        lower_b_err[j] = 0.0;
        upper_b_err[j] = 0.0;
        lower_nf[j] = false;
        upper_nf[j] = false;
    }

    for j in 0..num_outputs {
        for i in 0..num_neurons {
            let la = bounds.lower_a()[[j, i]];
            let ua = bounds.upper_a()[[j, i]];
            let alpha_lower_i = alpha_lo[i];
            let alpha_upper_i = alpha_up[i];
            let lam = lambda[i];
            let lam_int = lambda_int[i];

            // ---- Lower bound output: maximize lower ----
            if la > 0.0 {
                let product = la * alpha_lower_i;
                if product.is_finite() {
                    let stored = next_down_f32(product);
                    new_lower_a[[j, i]] = stored;
                    if track_err {
                        let e = in_lower_err.map_or(0.0, |m| m[[j, i]]);
                        if e != 0.0 {
                            let gap = ((la as f64) * (alpha_lower_i as f64) - stored as f64).abs();
                            let stable = (la as f64) > (e as f64);
                            new_lower_a_err[[j, i]] =
                                relu_alpha_coeff_err_val(e, stable, alpha_lower_i, lam, gap);
                            if !stable {
                                lower_b_err[j] =
                                    add_f64_up(lower_b_err[j], e as f64 * lam_int.abs() as f64);
                            }
                        }
                    }
                } else {
                    lower_nf[j] = true;
                }
            } else if la < 0.0 {
                let product = la * lam;
                if product.is_finite() {
                    let stored = next_down_f32(product);
                    new_lower_a[[j, i]] = stored;
                    if track_err {
                        let e = in_lower_err.map_or(0.0, |m| m[[j, i]]);
                        if e != 0.0 {
                            let gap = ((la as f64) * (lam as f64) - stored as f64).abs();
                            let stable = (-(la as f64)) > (e as f64);
                            new_lower_a_err[[j, i]] =
                                relu_alpha_coeff_err_val(e, stable, lam, alpha_lower_i, gap);
                            lower_b_err[j] =
                                add_f64_up(lower_b_err[j], e as f64 * lam_int.abs() as f64);
                        }
                    }
                } else {
                    lower_nf[j] = true;
                }
                new_lower_b_f64[j] = add_f64_down(new_lower_b_f64[j], la as f64 * lam_int as f64);
            } else if track_err && la == 0.0 {
                let e = in_lower_err.map_or(0.0, |m| m[[j, i]]);
                if e != 0.0 {
                    new_lower_a_err[[j, i]] =
                        relu_alpha_coeff_err_val(e, false, alpha_lower_i, lam, 0.0);
                    lower_b_err[j] = add_f64_up(lower_b_err[j], e as f64 * lam_int.abs() as f64);
                }
            }

            // ---- Upper bound output: minimize upper ----
            if ua > 0.0 {
                let product = ua * lam;
                if product.is_finite() {
                    let stored = next_up_f32(product);
                    new_upper_a[[j, i]] = stored;
                    if track_err {
                        let e = in_upper_err.map_or(0.0, |m| m[[j, i]]);
                        if e != 0.0 {
                            let gap = ((ua as f64) * (lam as f64) - stored as f64).abs();
                            let stable = (ua as f64) > (e as f64);
                            new_upper_a_err[[j, i]] =
                                relu_alpha_coeff_err_val(e, stable, lam, alpha_upper_i, gap);
                            upper_b_err[j] =
                                add_f64_up(upper_b_err[j], e as f64 * lam_int.abs() as f64);
                        }
                    }
                } else {
                    upper_nf[j] = true;
                }
                new_upper_b_f64[j] = add_f64_up(new_upper_b_f64[j], ua as f64 * lam_int as f64);
            } else if ua < 0.0 {
                let product = ua * alpha_upper_i;
                if product.is_finite() {
                    let stored = next_up_f32(product);
                    new_upper_a[[j, i]] = stored;
                    if track_err {
                        let e = in_upper_err.map_or(0.0, |m| m[[j, i]]);
                        if e != 0.0 {
                            let gap = ((ua as f64) * (alpha_upper_i as f64) - stored as f64).abs();
                            let stable = (-(ua as f64)) > (e as f64);
                            new_upper_a_err[[j, i]] =
                                relu_alpha_coeff_err_val(e, stable, alpha_upper_i, lam, gap);
                            if !stable {
                                upper_b_err[j] =
                                    add_f64_up(upper_b_err[j], e as f64 * lam_int.abs() as f64);
                            }
                        }
                    }
                } else {
                    upper_nf[j] = true;
                }
            } else if track_err && ua == 0.0 {
                let e = in_upper_err.map_or(0.0, |m| m[[j, i]]);
                if e != 0.0 {
                    new_upper_a_err[[j, i]] =
                        relu_alpha_coeff_err_val(e, false, lam, alpha_upper_i, 0.0);
                    upper_b_err[j] = add_f64_up(upper_b_err[j], e as f64 * lam_int.abs() as f64);
                }
            }
        }
    }

    if track_err {
        for j in 0..num_outputs {
            // Directed discharge, byte-for-byte the scalar arm's.
            new_lower_b_f64[j] = add_f64_down(new_lower_b_f64[j], -lower_b_err[j]);
            new_upper_b_f64[j] = add_f64_up(new_upper_b_f64[j], upper_b_err[j]);
        }
    }

    // Publish EXACTLY as `propagate_linear_with_alpha_impl` does, exact zero
    // preserved. This arm used to cast unconditionally, so a structural zero row
    // — every incoming coefficient and bias zero, the shape
    // `inject_stacked_identity_block` refuses on a nonzero find — published
    // `-1.4e-45` here and `0.0` from the scalar twin: a bit-parity break in the
    // one function whose entire contract is bit-identity. The zero branch is
    // sound for the reason given at the scalar publication site: with directed
    // steps an accumulator reaches exact `+0.0` only from all-zero terms or from
    // a step down off `+5e-324`.
    let mut new_lower_b = new_lower_b_f64.mapv(|v| {
        if v == 0.0 {
            0.0
        } else {
            next_down_f32(v as f32)
        }
    });
    let mut new_upper_b =
        new_upper_b_f64.mapv(|v| if v == 0.0 { 0.0 } else { next_up_f32(v as f32) });
    for j in 0..num_outputs {
        if lower_nf[j] {
            for i in 0..num_neurons {
                new_lower_a[[j, i]] = 0.0;
                if track_err {
                    new_lower_a_err[[j, i]] = 0.0;
                }
            }
            new_lower_b[j] = f32::NEG_INFINITY;
        }
        if upper_nf[j] {
            for i in 0..num_neurons {
                new_upper_a[[j, i]] = 0.0;
                if track_err {
                    new_upper_a_err[[j, i]] = 0.0;
                }
            }
            new_upper_b[j] = f32::INFINITY;
        }
    }

    let out = if track_err {
        let mut b = LinearBounds::new_or_conservative_with_err(
            new_lower_a,
            new_lower_b,
            new_upper_a,
            new_upper_b,
            new_lower_a_err,
            new_upper_a_err,
        )?;
        // LOCAL DISCHARGE over THIS layer's pre-activation box (matches the scalar
        // path: contiguous pre_lower/pre_upper always fold here).
        b.fold_coeff_err_into_bias(pre_lower, pre_upper);
        b
    } else {
        LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)?
    };
    Ok(out)
}

/// One domain of the heuristic (default-α) ReLU backward — a faithful transcription
/// of `crown_elementwise_backward(bounds, pre, relu_linear_relaxation)` (finite
/// pre-activation only), reusing the shared [`compose`] primitives so the composed
/// coefficients and the ALWAYS-carried directed-rounding-gap error are bit-identical.
///
/// Its reference lives in ANOTHER FILE (`layers/common/crown_dense.rs`), so every
/// certificate change there must land here in the same commit. It did not, once:
/// certifying the dense twin's covers with `cover_sum_up` while this transcription
/// still summed them in binary32 moved `upper_a_err` by 1 ULP and broke
/// `test_batched_relu_bit_identical_to_per_domain_kernel_lsnc_step2`.
#[allow(clippy::too_many_arguments)]
fn relu_batched_heuristic_domain(
    bounds: &LinearBounds,
    pre_lower: &[f32],
    pre_upper: &[f32],
    num_outputs: usize,
    num_neurons: usize,
    lower_b_err: &mut [f64],
    upper_b_err: &mut [f64],
    lower_nf: &mut [bool],
    upper_nf: &mut [bool],
) -> Result<LinearBounds> {
    use crate::layers::common::compose;

    // Per-neuron relaxations (finite l,u → identity/zero/crossing arms only).
    let relaxations: Vec<LinearRelaxation> = (0..num_neurons)
        .map(|i| relu_linear_relaxation(pre_lower[i], pre_upper[i]))
        .collect();

    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    // Certified constant + exact seed decode (#vnncomp-aw-constant), byte-for-byte
    // the certified `crown_elementwise_backward_indexed` this function
    // transcribes — see the rationale above that function's accumulators. The
    // 1-ULP `upper_a_err` divergence the kernel-parity test caught was this arm
    // still summing its two-line covers in binary32 after the dense twin had
    // been certified; the covers below now use the same `cover_sum_up`.
    let mut new_lower_b_f64 = bounds.lower_b().mapv(f32_to_f64_exact);
    let mut new_upper_b_f64 = bounds.upper_b().mapv(f32_to_f64_exact);

    let in_lower_err = bounds.lower_a_err();
    let in_upper_err = bounds.upper_a_err();
    let mut new_lower_a_err = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_upper_a_err = Array2::<f32>::zeros((num_outputs, num_neurons));
    for j in 0..num_outputs {
        lower_b_err[j] = 0.0;
        upper_b_err[j] = 0.0;
        lower_nf[j] = false;
        upper_nf[j] = false;
    }

    // Slope actually used by compose_lower / compose_upper for a coefficient's sign.
    let lr_slope = |a: f32, relax: &LinearRelaxation| -> f64 {
        if a > 0.0 {
            relax.lower_slope as f64
        } else if a < 0.0 {
            relax.upper_slope as f64
        } else {
            0.0
        }
    };
    let ur_slope = |a: f32, relax: &LinearRelaxation| -> f64 {
        if a > 0.0 {
            relax.upper_slope as f64
        } else if a < 0.0 {
            relax.lower_slope as f64
        } else {
            0.0
        }
    };

    for j in 0..num_outputs {
        for i in 0..num_neurons {
            let la = bounds.lower_a()[[j, i]];
            let ua = bounds.upper_a()[[j, i]];
            let relax = &relaxations[i];

            let lr = compose::compose_lower(la, relax);
            new_lower_a[[j, i]] = lr.new_coeff;
            // Certified accumulation (#vnncomp-aw-constant), identical to the
            // dense twin's.
            new_lower_b_f64[j] = add_f64_down(new_lower_b_f64[j], lr.intercept_contrib);
            lower_nf[j] |= lr.nonfinite;

            let ur = compose::compose_upper(ua, relax);
            new_upper_a[[j, i]] = ur.new_coeff;
            new_upper_b_f64[j] = add_f64_up(new_upper_b_f64[j], ur.intercept_contrib);
            upper_nf[j] |= ur.nonfinite;

            // Outward two-line covers (see `cover_sum_up`). The dense twin
            // computes these with the identical helper; summing them in binary32
            // here is what broke `upper_a_err` bit-parity by 1 ULP.
            let slope_sum = cover_sum_up(relax.lower_slope, relax.upper_slope);
            let int_sum = cover_sum_up(relax.lower_intercept, relax.upper_intercept);

            {
                let gap = if la != 0.0 {
                    (la as f64 * lr_slope(la, relax) - lr.new_coeff as f64).abs()
                } else {
                    0.0
                };
                let ea = in_lower_err.map_or(0.0, |e| e[[j, i]] as f64);
                if gap != 0.0 || ea != 0.0 {
                    let (slope_cover, int_cover) = if (la as f64).abs() > ea {
                        let slope = lr_slope(la, relax).abs();
                        let intercept = if la > 0.0 {
                            relax.lower_intercept.abs() as f64
                        } else {
                            relax.upper_intercept.abs() as f64
                        };
                        (slope, intercept)
                    } else {
                        (slope_sum, int_sum)
                    };
                    new_lower_a_err[[j, i]] = next_up_f32((ea * slope_cover + gap) as f32);
                    if ea != 0.0 {
                        // Non-negative charge: `add_f64_up`, since a
                        // round-to-nearest `+=` UNDER-counts a widening.
                        lower_b_err[j] = add_f64_up(lower_b_err[j], ea * int_cover);
                    }
                }
            }

            {
                let gap = if ua != 0.0 {
                    (ua as f64 * ur_slope(ua, relax) - ur.new_coeff as f64).abs()
                } else {
                    0.0
                };
                let ea = in_upper_err.map_or(0.0, |e| e[[j, i]] as f64);
                if gap != 0.0 || ea != 0.0 {
                    let (slope_cover, int_cover) = if (ua as f64).abs() > ea {
                        let slope = ur_slope(ua, relax).abs();
                        let intercept = if ua > 0.0 {
                            relax.upper_intercept.abs() as f64
                        } else {
                            relax.lower_intercept.abs() as f64
                        };
                        (slope, intercept)
                    } else {
                        (slope_sum, int_sum)
                    };
                    new_upper_a_err[[j, i]] = next_up_f32((ea * slope_cover + gap) as f32);
                    if ea != 0.0 {
                        upper_b_err[j] = add_f64_up(upper_b_err[j], ea * int_cover);
                    }
                }
            }
        }
        // Directed discharge — the most cancellation-prone add in the row.
        new_lower_b_f64[j] = add_f64_down(new_lower_b_f64[j], -lower_b_err[j]);
        new_upper_b_f64[j] = add_f64_up(new_upper_b_f64[j], upper_b_err[j]);
    }

    // Publish EXACTLY as `crown_elementwise_backward_indexed` does, exact zero
    // preserved. This arm used to cast unconditionally, so a structural zero row
    // published `-1.4e-45` here and `0.0` from the dense reference — a latent
    // bit-parity break in the one function whose entire contract is bit-identity
    // (the kernel-parity test's random biases never land on exact zero, so it
    // could not see this one). Sound for the reason given at the dense
    // publication site: with directed steps an accumulator reaches exact `+0.0`
    // only from all-zero terms or from a step down off `+5e-324`.
    let mut new_lower_b = new_lower_b_f64.mapv(|x| {
        if x == 0.0 {
            0.0
        } else {
            next_down_f32(x as f32)
        }
    });
    let mut new_upper_b =
        new_upper_b_f64.mapv(|x| if x == 0.0 { 0.0 } else { next_up_f32(x as f32) });
    for j in 0..num_outputs {
        if lower_nf[j] {
            for i in 0..num_neurons {
                new_lower_a[[j, i]] = 0.0;
                new_lower_a_err[[j, i]] = 0.0;
            }
            new_lower_b[j] = f32::NEG_INFINITY;
        }
        if upper_nf[j] {
            for i in 0..num_neurons {
                new_upper_a[[j, i]] = 0.0;
                new_upper_a_err[[j, i]] = 0.0;
            }
            new_upper_b[j] = f32::INFINITY;
        }
    }

    LinearBounds::new_or_conservative_with_err(
        new_lower_a,
        new_lower_b,
        new_upper_a,
        new_upper_b,
        new_lower_a_err,
        new_upper_a_err,
    )
}

impl BoundPropagation for ReLULayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        relu_ibp(input)
    }

    /// CROWN backward propagation requires pre-activation bounds.
    /// Use `ReLULayer::propagate_linear_with_bounds` instead.
    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::InvalidSpec(
            "ReLU CROWN propagation requires pre-activation bounds. \
             Use propagate_linear_with_bounds() instead."
                .to_string(),
        ))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        // Delegate to the inherent method.
        ReLULayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl ReLULayer {
    /// Cooperative finite-deadline Patches backward for materialized 7D
    /// explicit rows.
    ///
    /// This is intentionally an inherent, ReLU-only face rather than a change
    /// to [`crate::layers::common::PatchesPropagation`]: the default-dark
    /// Spec-CROWN call site is the sole production caller, while every other
    /// activation and every no-deadline ReLU keeps the historical trait path.
    pub(crate) fn propagate_patches_with_bounds_and_deadline(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
        deadline: std::time::Instant,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        // NaN-only guard (see the trait implementation below): the patches
        // path uses the same proven relu_linear_relaxation infinite branches.
        nan_only_domain_guard("ReLU", pre_activation)?;
        crate::layers::common::crown_elementwise_backward_patches_with_deadline(
            bounds,
            pre_activation,
            deadline,
            relu_linear_relaxation,
        )
    }

    /// CROWN backward through ReLU in Patches mode with optimizable alpha.
    ///
    /// This is the Patches-mode counterpart of [`Self::propagate_linear_with_alpha`].
    /// Uses `alpha[i]` as the lower bound slope for crossing neurons instead of the
    /// heuristic value. Returns `(CrownBounds, gradient)` where `gradient[i]` =
    /// `d(sum of lower bounds)/d(alpha[i])`.
    ///
    /// Reference: alpha-beta-CROWN auto_LiRPA/operators/relu.py (Patches alpha backward)
    pub(crate) fn propagate_patches_with_alpha(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
        alpha: &Array1<f32>,
    ) -> Result<(crate::bounds::patches::CrownBounds, Array1<f32>)> {
        non_finite_domain_guard("ReLU-alpha-patches", pre_activation)?;
        crate::layers::common::crown_relu_backward_patches_with_alpha(bounds, pre_activation, alpha)
    }

    /// Bound-only counterpart of [`Self::propagate_patches_with_alpha`].
    pub(crate) fn propagate_patches_with_alpha_bound_only(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
        alpha: &Array1<f32>,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("ReLU-alpha-patches", pre_activation)?;
        crate::layers::common::crown_relu_backward_patches_with_alpha_bound_only(
            bounds,
            pre_activation,
            alpha,
        )
    }

    /// Prevalidate the narrowly admitted owned, in-place alpha-ReLU Patches route.
    ///
    /// This must run while the carrier is still borrowed. A successful plan
    /// certifies that every fallible layout/shape/alpha/error invariant was
    /// checked before the caller transfers ownership.
    pub(crate) fn prepare_patches_with_alpha_in_place(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
        alpha: &Array1<f32>,
        deadline: std::time::Instant,
    ) -> Result<crate::layers::common::PreparedAlphaPatchesReluInPlace> {
        crate::layers::common::prepare_crown_relu_backward_patches_with_alpha_in_place(
            bounds,
            pre_activation,
            alpha,
            deadline,
        )
    }

    /// Execute a prepared owned alpha-ReLU transform in place.
    ///
    /// On deadline expiry the owned carrier is dropped. The caller must
    /// propagate the error upward and must not enter a Dense/Patches fallback.
    pub(crate) fn propagate_prepared_patches_with_alpha_in_place(
        &self,
        bounds: Box<crate::bounds::patches::PatchesLinearBounds>,
        prepared: crate::layers::common::PreparedAlphaPatchesReluInPlace,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        crate::layers::common::crown_relu_backward_patches_with_alpha_in_place(
            bounds,
            prepared,
            pre_activation,
        )
    }
}

impl crate::layers::common::PatchesPropagation for ReLULayer {
    fn propagate_patches(
        &self,
        _bounds: &crate::bounds::patches::PatchesLinearBounds,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        Err(NyError::InvalidSpec(
            "ReLU Patches propagation requires pre-activation bounds. \
             Use propagate_patches_with_bounds() instead."
                .to_string(),
        ))
    }

    /// CROWN backward through ReLU in Patches mode.
    ///
    /// Scales patches coefficients by per-neuron ReLU relaxation slopes,
    /// selecting lower vs upper relaxation based on coefficient sign.
    /// This keeps bounds in Patches form, avoiding the O(n²) Dense materialization
    /// that would occur via ensure_dense() fallback.
    ///
    /// Reference: alpha-beta-CROWN auto_LiRPA/operators/relu.py (Patches backward)
    fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        // NaN-only guard (see propagate_linear_with_bounds): the patches path uses the
        // same proven relu_linear_relaxation infinite-case branches.
        nan_only_domain_guard("ReLU", pre_activation)?;
        crate::layers::common::crown_elementwise_backward_patches(
            bounds,
            pre_activation,
            relu_linear_relaxation,
        )
    }
}

/// Zero-copy pre-activation reads in the two dense CROWN backward entries.
///
/// Every case asserts the change's OWN OBSERVABLE — `flat_reads`' borrow/copy
/// counters — and not only that two sets of floats agree. Reinstating
/// `BoundedTensor::flatten()` would still produce equal numbers, so an
/// equality-only test would pass while the per-call copies came back; the
/// counter delta is what fails closed.
#[cfg(test)]
mod zero_copy_pre_activation_tests {
    use super::*;
    use ndarray::IxDyn;

    /// Finite ramp straddling zero, so the crossing and always-active arms
    /// both run within a single call.
    fn ramp(len: usize) -> Vec<f32> {
        (0..len).map(|i| i as f32 * 0.25 - 1.0).collect()
    }

    /// A `[3, 2]` row-major source with reversed axes: logically `[2, 3]` and
    /// NOT contiguous (the construction `util::contiguous`'s own test uses).
    fn reversed_axes_owned(values: Vec<f32>) -> ArrayD<f32> {
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), values)
            .expect("valid shape")
            .view()
            .reversed_axes()
            .to_owned()
    }

    /// The same values in the same ROW-MAJOR LOGICAL order that
    /// [`reversed_axes_owned`] presents: element `(i, j)` of the reversed
    /// `[2, 3]` view is `src[j * 2 + i]`.
    fn logical_order(v: &[f32]) -> Vec<f32> {
        vec![v[0], v[2], v[4], v[1], v[3], v[5]]
    }

    fn assert_bits_eq_2d(actual: &Array2<f32>, expected: &Array2<f32>, label: &str) {
        assert_eq!(actual.shape(), expected.shape(), "{label}: shape");
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}[{idx}]: {a} is not bit-identical to {e}"
            );
        }
    }

    fn assert_bits_eq_1d(actual: &Array1<f32>, expected: &Array1<f32>, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: len");
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}[{idx}]: {a} is not bit-identical to {e}"
            );
        }
    }

    /// A standard-layout pre-activation — the layout every production ReLU
    /// node arrives in — must be read by BORROW at both endpoints, and a
    /// rank-2 endpoint must give bit-identical bounds to the rank-1 endpoint
    /// holding the same values (the equivalence the removed `flatten()` used
    /// to establish by copying).
    #[ntest::timeout(5000)]
    #[test]
    fn alpha_backward_borrows_standard_layout_pre_activation() -> Result<()> {
        let layer = ReLULayer::new();
        let lower_2d = ArrayD::from_shape_vec(IxDyn(&[2, 3]), ramp(6)).expect("valid shape");
        let upper_2d = lower_2d.mapv(|v| v + 2.0);
        assert!(
            lower_2d.as_slice().is_some() && upper_2d.as_slice().is_some(),
            "precondition: a freshly built row-major endpoint is standard layout"
        );
        let pre_2d = BoundedTensor::new(lower_2d, upper_2d)?;
        let pre_1d = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[6]), ramp(6)).expect("valid shape"),
            ArrayD::from_shape_vec(IxDyn(&[6]), ramp(6).iter().map(|v| v + 2.0).collect())
                .expect("valid shape"),
        )?;
        let bounds = LinearBounds::identity(6);
        let alpha = Array1::<f32>::from_elem(6, 0.5);

        let before = flat_reads::snapshot();
        let (out_2d, gl_2d, gu_2d) =
            layer.propagate_linear_with_alpha(&bounds, &pre_2d, &alpha, None)?;
        let mid = flat_reads::snapshot();
        let (out_1d, gl_1d, gu_1d) =
            layer.propagate_linear_with_alpha(&bounds, &pre_1d, &alpha, None)?;
        let after = flat_reads::snapshot();

        assert_eq!(
            (mid.0 - before.0, mid.1 - before.1),
            (2, 0),
            "rank-2 standard-layout endpoints must both be BORROWED, \
             i.e. 2 borrowed / 0 copied; got {before:?} -> {mid:?}"
        );
        assert_eq!(
            (after.0 - mid.0, after.1 - mid.1),
            (2, 0),
            "rank-1 standard-layout endpoints must both be BORROWED, \
             i.e. 2 borrowed / 0 copied; got {mid:?} -> {after:?}"
        );

        assert_bits_eq_2d(out_2d.lower_a(), out_1d.lower_a(), "lower_a");
        assert_bits_eq_2d(out_2d.upper_a(), out_1d.upper_a(), "upper_a");
        assert_bits_eq_1d(out_2d.lower_b(), out_1d.lower_b(), "lower_b");
        assert_bits_eq_1d(out_2d.upper_b(), out_1d.upper_b(), "upper_b");
        assert_bits_eq_1d(&gl_2d, &gl_1d, "gradient_lower");
        assert_bits_eq_1d(&gu_2d, &gu_1d, "gradient_upper");
        Ok(())
    }

    /// A non-standard-layout endpoint takes the COPY arm — the fallback that
    /// reproduces exactly what `flatten()` used to do unconditionally — and
    /// must yield bit-identical bounds to the same values laid out
    /// contiguously.
    #[ntest::timeout(5000)]
    #[test]
    fn alpha_backward_non_contiguous_pre_activation_is_bit_identical() -> Result<()> {
        let layer = ReLULayer::new();
        let src_lower = ramp(6);
        let src_upper: Vec<f32> = src_lower.iter().map(|v| v + 2.0).collect();
        let lower_nc = reversed_axes_owned(src_lower.clone());
        let upper_nc = reversed_axes_owned(src_upper.clone());
        assert!(
            lower_nc.as_slice().is_none() && upper_nc.as_slice().is_none(),
            "precondition: reversed-axes endpoints must be non-contiguous"
        );

        let pre_nc = BoundedTensor::new(lower_nc, upper_nc)?;
        // The contiguous control: same values, same logical order.
        let pre_c = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), logical_order(&src_lower)).expect("valid shape"),
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), logical_order(&src_upper)).expect("valid shape"),
        )?;
        let bounds = LinearBounds::identity(6);
        let alpha = Array1::<f32>::from_elem(6, 0.5);

        let before = flat_reads::snapshot();
        let (out_nc, gl_nc, gu_nc) =
            layer.propagate_linear_with_alpha(&bounds, &pre_nc, &alpha, None)?;
        let mid = flat_reads::snapshot();
        let (out_c, gl_c, gu_c) =
            layer.propagate_linear_with_alpha(&bounds, &pre_c, &alpha, None)?;
        let after = flat_reads::snapshot();

        assert_eq!(
            (mid.0 - before.0, mid.1 - before.1),
            (0, 2),
            "non-contiguous endpoints must take the COPY arm, \
             i.e. 0 borrowed / 2 copied; got {before:?} -> {mid:?}"
        );
        assert_eq!(
            (after.0 - mid.0, after.1 - mid.1),
            (2, 0),
            "the contiguous control must take the BORROW arm, \
             i.e. 2 borrowed / 0 copied; got {mid:?} -> {after:?}"
        );

        assert_bits_eq_2d(out_nc.lower_a(), out_c.lower_a(), "lower_a");
        assert_bits_eq_2d(out_nc.upper_a(), out_c.upper_a(), "upper_a");
        assert_bits_eq_1d(out_nc.lower_b(), out_c.lower_b(), "lower_b");
        assert_bits_eq_1d(out_nc.upper_b(), out_c.upper_b(), "upper_b");
        assert_bits_eq_1d(&gl_nc, &gl_c, "gradient_lower");
        assert_bits_eq_1d(&gu_nc, &gu_c, "gradient_upper");
        Ok(())
    }

    /// The SDP-CROWN entry carried the identical read, so it gets the
    /// identical proof — its own `flatten()` + `clone()` chain is gone too.
    /// `x_hat` keeps its separate `contiguous_flat_slice` call, which is not
    /// a pre-activation read and so must not move these counters.
    #[ntest::timeout(5000)]
    #[test]
    fn sdp_backward_borrows_standard_layout_pre_activation() -> Result<()> {
        let layer = ReLULayer::new();
        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), ramp(4)).expect("valid shape");
        let upper = lower.mapv(|v| v + 2.0);
        let pre = BoundedTensor::new(lower, upper)?;
        let bounds = LinearBounds::identity(4);
        let x_hat = Array1::<f32>::zeros(4);

        let before = flat_reads::snapshot();
        let _out = layer.propagate_linear_with_bounds_sdp(&bounds, &pre, &x_hat, 0.5)?;
        let after = flat_reads::snapshot();

        assert_eq!(
            (after.0 - before.0, after.1 - before.1),
            (2, 0),
            "SDP-CROWN backward must BORROW both standard-layout endpoints, \
             i.e. 2 borrowed / 0 copied; got {before:?} -> {after:?}"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests;

/// Cancelling-constant certificate for the α-CROWN ReLU dense backward and for
/// both domain-batched arms (#vnncomp-aw-constant).
///
/// These tests FAIL on the pre-fix code, where every bias accumulator here was a
/// bare round-to-nearest binary64 `+=`. They are named individually so a filter
/// that stops matching them fails loudly rather than silently passing zero tests
/// (`cargo test <filter>` exits 0 on an empty match).
///
/// The shape is the one the ReLU backward can actually reach: the lower constant
/// accumulator only ever receives `la * lambda_intercept` with `la < 0` and
/// `lambda_intercept >= 0`, i.e. only NEGATIVE terms, so cancellation cannot come
/// from the terms alone — it comes from the SEED. A seed of `+2^60 * i` (a bias
/// arriving from the layer above) followed by `UNITS` terms of `-i` and one term
/// of `-2^60 * i` is exactly that: every unit term vanishes into the seed's
/// 256-wide binary64 ulp, the giant term then cancels the seed exactly, and the
/// publication's exact-zero branch hands back `0.0` where the truth is
/// `-UNITS * i`. The upper direction is the mirror image (`ua > 0`, seed
/// `-2^60 * i`).
#[cfg(test)]
mod certified_constant_tests {
    use super::*;

    /// `2^60`, exactly representable in binary32.
    const BIG: f32 = 1_152_921_504_606_846_976.0;
    /// How many unit terms vanish into the seed.
    const UNITS: usize = 4096;
    /// `UNITS` unit neurons, then the one giant neuron that cancels the seed.
    const N: usize = UNITS + 1;
    /// Crossing pre-activation box (`l < 0 < u`) so every neuron takes the
    /// triangle arm and carries a NONZERO upper-chord intercept.
    const L: f32 = -2.0;
    const U: f32 = 2.0;

    /// The upper-chord intercept each neuron contributes, read from the
    /// production helper rather than hard-coded: `relu_crossing_upper_chord`
    /// rounds both chord parameters outward, so the intercept is NOT the tidy
    /// `1.0` the geometry suggests and a hard-coded constant would silently stop
    /// describing the code under test.
    fn chord_intercept() -> f32 {
        let (lambda, intercept) = relu_crossing_upper_chord(L, U, None);
        assert!(
            lambda.is_finite() && intercept.is_finite() && intercept > 0.0,
            "test setup: a finite crossing box must yield a finite positive \
             upper-chord intercept, got lambda={lambda} intercept={intercept}"
        );
        intercept
    }

    /// `2^60 * intercept`, which is exactly representable in binary32 (a power of
    /// two times a 24-bit significand). The assertion is load-bearing: if the
    /// seed were not EXACTLY the negation of the giant term, the pre-fix
    /// accumulator would not land on exact zero and the counterexample would
    /// prove nothing.
    fn seed_constant(intercept: f32) -> f32 {
        let exact = BIG as f64 * intercept as f64;
        let seed = exact as f32;
        assert_eq!(
            f32_to_f64_exact(seed),
            exact,
            "test setup: the seed must equal 2^60 * intercept EXACTLY"
        );
        seed
    }

    /// One `[1, N]` cancelling row in BOTH directions, plus the crossing box.
    fn cancelling_inputs() -> (LinearBounds, BoundedTensor, f32) {
        let intercept = chord_intercept();
        let seed = seed_constant(intercept);
        let lower_a = Array2::from_shape_fn((1, N), |(_, i)| if i == N - 1 { -BIG } else { -1.0 });
        let upper_a = Array2::from_shape_fn((1, N), |(_, i)| if i == N - 1 { BIG } else { 1.0 });
        let bounds = LinearBounds::new(
            lower_a,
            Array1::from_elem(1, seed),
            upper_a,
            Array1::from_elem(1, -seed),
        )
        .expect("finite coefficients and biases");
        let pre = BoundedTensor::new(
            Array1::from_elem(N, L).into_dyn(),
            Array1::from_elem(N, U).into_dyn(),
        )
        .expect("matching 1-D pre-activation shapes");
        (bounds, pre, intercept)
    }

    /// Both directions, one call. Widening the accumulator to binary64 is not a
    /// certificate: the exact biases are `∓UNITS * intercept` and a bare `+=`
    /// lands on exactly `0.0`.
    fn assert_certified(lower_b: &Array1<f32>, upper_b: &Array1<f32>, intercept: f32, arm: &str) {
        // `UNITS` is a power of two and `intercept` has 24 significand bits, so
        // both products are EXACT in binary64.
        let true_lower = -(UNITS as f64) * intercept as f64;
        let true_upper = (UNITS as f64) * intercept as f64;
        assert!(
            f32_to_f64_exact(lower_b[0]) <= true_lower,
            "{arm}: lower bias {} is ABOVE its exact value {true_lower} — a false \
             proof by {} (cancellation ate the binary64 accumulation error and the \
             binary32 ULP of the result cannot pay for it)",
            lower_b[0],
            f32_to_f64_exact(lower_b[0]) - true_lower
        );
        assert!(
            f32_to_f64_exact(upper_b[0]) >= true_upper,
            "{arm}: upper bias {} is BELOW its exact value {true_upper} — a false \
             proof by {}",
            upper_b[0],
            true_upper - f32_to_f64_exact(upper_b[0])
        );
        assert!(
            lower_b[0].is_finite() && upper_b[0].is_finite(),
            "{arm}: the certificate must loosen the bias, not vacate the row: \
             lower={} upper={}",
            lower_b[0],
            upper_b[0]
        );
    }

    fn assert_bits_eq_1d(actual: &Array1<f32>, expected: &Array1<f32>, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: len");
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}[{idx}]: {a} is not bit-identical to {e}"
            );
        }
    }

    fn assert_bits_eq_2d(actual: &Array2<f32>, expected: &Array2<f32>, label: &str) {
        assert_eq!(actual.shape(), expected.shape(), "{label}: shape");
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}[{idx}]: {a} is not bit-identical to {e}"
            );
        }
    }

    /// Every published plane, including the certified coefficient-error planes —
    /// the ones the kernel-parity test caught diverging by 1 ULP.
    fn assert_bounds_bit_identical(actual: &LinearBounds, expected: &LinearBounds, label: &str) {
        assert_bits_eq_2d(
            actual.lower_a(),
            expected.lower_a(),
            &format!("{label} lower_a"),
        );
        assert_bits_eq_2d(
            actual.upper_a(),
            expected.upper_a(),
            &format!("{label} upper_a"),
        );
        assert_bits_eq_1d(
            actual.lower_b(),
            expected.lower_b(),
            &format!("{label} lower_b"),
        );
        assert_bits_eq_1d(
            actual.upper_b(),
            expected.upper_b(),
            &format!("{label} upper_b"),
        );
        match (actual.lower_a_err(), expected.lower_a_err()) {
            (Some(a), Some(e)) => assert_bits_eq_2d(a, e, &format!("{label} lower_a_err")),
            (None, None) => {}
            (a, e) => panic!(
                "{label}: lower_a_err PRESENCE differs (actual={} expected={}) — an \
                 err plane that exists on one arm only is a certificate difference, \
                 not a formatting one",
                a.is_some(),
                e.is_some()
            ),
        }
        match (actual.upper_a_err(), expected.upper_a_err()) {
            (Some(a), Some(e)) => assert_bits_eq_2d(a, e, &format!("{label} upper_a_err")),
            (None, None) => {}
            (a, e) => panic!(
                "{label}: upper_a_err PRESENCE differs (actual={} expected={})",
                a.is_some(),
                e.is_some()
            ),
        }
    }

    fn dual_alphas() -> (Array1<f32>, Array1<f32>) {
        (
            Array1::<f32>::from_elem(N, 0.5),
            Array1::<f32>::from_elem(N, 0.25),
        )
    }

    /// The α-CROWN dense backward itself — `propagate_linear_with_alpha`, the
    /// per-domain reference the input-split ReLU arm calls.
    #[ntest::timeout(60000)]
    #[test]
    fn alpha_dense_backward_certifies_cancelling_constant_in_both_directions() {
        let (bounds, pre, intercept) = cancelling_inputs();
        let (alpha, alpha_upper) = dual_alphas();
        let (out, _gradient_lower, _gradient_upper) = ReLULayer::new()
            .propagate_linear_with_alpha(&bounds, &pre, &alpha, Some(&alpha_upper))
            .expect("a finite crossing relaxation must not error");
        assert_certified(out.lower_b(), out.upper_b(), intercept, "alpha dense");
    }

    /// The domain-batched α arm must be certified AND bit-identical to the scalar
    /// α arm it transcribes — both live in this file, so a divergence here is a
    /// fourth idiom, not a cross-file drift.
    #[ntest::timeout(60000)]
    #[test]
    fn batched_alpha_domain_certifies_and_matches_scalar_alpha_bit_for_bit() {
        let (bounds, pre, intercept) = cancelling_inputs();
        let (alpha, alpha_upper) = dual_alphas();
        let relu = ReLULayer::new();
        let reference = relu
            .propagate_linear_with_alpha(&bounds, &pre, &alpha, Some(&alpha_upper))
            .expect("scalar alpha reference")
            .0;
        let batched = relu
            .propagate_linear_multi_domain_relu(&[&bounds], &[&pre], &[Some((alpha, alpha_upper))])
            .expect("batched relu ok")
            .expect("finite contiguous inputs must not decline");
        assert_eq!(batched.len(), 1, "one domain in, one domain out");
        assert_certified(
            batched[0].lower_b(),
            batched[0].upper_b(),
            intercept,
            "batched alpha domain",
        );
        assert_bounds_bit_identical(&batched[0], &reference, "batched alpha vs scalar alpha:");
    }

    /// The domain-batched HEURISTIC arm must be certified AND bit-identical to
    /// `propagate_linear_with_bounds` — whose certificate lives in ANOTHER file
    /// (`layers/common/crown_dense.rs`). This is the pair the kernel-parity test
    /// caught diverging by 1 ULP in `upper_a_err`.
    #[ntest::timeout(60000)]
    #[test]
    fn batched_heuristic_domain_certifies_and_matches_dense_reference_bit_for_bit() {
        let (bounds, pre, intercept) = cancelling_inputs();
        let relu = ReLULayer::new();
        let reference = relu
            .propagate_linear_with_bounds(&bounds, &pre)
            .expect("dense heuristic reference");
        let batched = relu
            .propagate_linear_multi_domain_relu(&[&bounds], &[&pre], &[None])
            .expect("batched relu ok")
            .expect("finite contiguous inputs must not decline");
        assert_eq!(batched.len(), 1, "one domain in, one domain out");
        assert_certified(
            batched[0].lower_b(),
            batched[0].upper_b(),
            intercept,
            "batched heuristic domain",
        );
        assert_certified(
            reference.lower_b(),
            reference.upper_b(),
            intercept,
            "dense heuristic reference",
        );
        assert_bounds_bit_identical(
            &batched[0],
            &reference,
            "batched heuristic vs crown_dense reference:",
        );
    }

    /// The two-line cover must never round INWARD. `cover_sum_up` is the helper
    /// the sign-ambiguous error carry rests on in all three arms of this file;
    /// the bare `(a.abs() + b.abs()) as f64` it replaced performs the addition in
    /// BINARY32 and lands below the true sum whenever the two operands differ
    /// enough in exponent.
    #[ntest::timeout(5000)]
    #[test]
    fn cover_sum_up_never_rounds_the_two_line_cover_inward() {
        // 1.0 + 2^-30: the exact sum needs 31 significand bits, so binary32
        // round-to-nearest returns exactly 1.0 — BELOW the truth.
        let a = 1.0f32;
        let b = f32::from_bits(0x3080_0000); // 2^-30
        assert_eq!(
            (a.abs() + b.abs()) as f64,
            1.0f64,
            "test setup: the bare binary32 form must actually round inward here"
        );
        let exact = f32_to_f64_exact(a) + f32_to_f64_exact(b);
        assert!(
            cover_sum_up(a, b) >= exact,
            "cover_sum_up({a}, {b}) = {} is BELOW the exact sum {exact}",
            cover_sum_up(a, b)
        );
        // Two zero intercepts contribute exactly nothing: the cover must stay an
        // exact 0.0 rather than smearing a 5e-324 charge across every cell.
        assert_eq!(cover_sum_up(0.0, -0.0), 0.0);
        // Signs are irrelevant to a cover of magnitudes.
        assert_eq!(cover_sum_up(-a, b), cover_sum_up(a, -b));
    }
}
