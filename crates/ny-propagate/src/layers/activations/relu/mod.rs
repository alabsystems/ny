// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2};
use ny_core::{nan_propagating_max, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;
use tracing::debug;

use super::{relu_crossing_upper_chord, LinearRelaxation};
use crate::layers::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched, nan_only_domain_guard,
    non_finite_domain_guard, BoundPropagation,
};
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
/// Made pub(crate) for GPU CROWN layer descriptor extraction (#3397).
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

        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower().shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper().shape().to_vec(),
            })?;

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
                // Guard: skip zero coefficients to avoid 0*inf NaN (#1739).
                // Directed rounding on slope products (#2786): next_down_f32 for lower_a,
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
    /// This path intentionally remains custom after #1787 refactoring: the canonical
    /// `crown_elementwise_backward` helper does not expose per-neuron gradient
    /// accumulation for α parameters.
    ///
    /// # Dual Alpha (#3393)
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
        self.propagate_linear_with_alpha_impl(bounds, pre_activation, alpha, alpha_upper, true)
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
        self.propagate_linear_with_alpha_impl(bounds, pre_activation, alpha, alpha_upper, false)
            .map(|(bounds, _, _)| bounds)
    }

    fn propagate_linear_with_alpha_impl(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        alpha: &Array1<f32>,
        alpha_upper: Option<&Array1<f32>>,
        track_gradients: bool,
    ) -> Result<(LinearBounds, Array1<f32>, Array1<f32>)> {
        debug!("ReLU layer α-CROWN backward propagation");
        // Task #35 reachability audit: does any production path reach here with
        // NONZERO incoming certified coefficient error (which this fn drops)?
        alpha_err_audit::record(bounds);
        non_finite_domain_guard("ReLU-alpha", pre_activation)?;

        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower().shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper().shape().to_vec(),
            })?;

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
        // Match canonical helper precision: accumulate bias in f64 to reduce cancellation.
        let mut new_lower_b_f64 = bounds.lower_b().mapv(|v| v as f64);
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b_f64 = bounds.upper_b().mapv(|v| v as f64);

        // Track which output rows have non-finite coefficients after coeff × slope (#3086).
        // When accumulated Inf coefficients multiply slopes, f32 can produce Inf/NaN.
        // Affected rows get zeroed A-coefficients and ±Inf bias (sound conservative fallback),
        // preserving precision in all other rows. Matches canonical pattern from
        // crown_elementwise_backward_indexed (#3009) and crown_single (#2681).
        let mut lower_nonfinite_rows = vec![false; num_outputs];
        let mut upper_nonfinite_rows = vec![false; num_outputs];

        // Dual alpha (#3393): separate gradients for lower and upper bound paths.
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
        // When the incoming bounds carry NO error, this whole block is skipped and
        // the result is byte-identical to the pre-#35 code (no err attached).
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
        // added to upper_b (outward), accumulated in f64.
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
                e64 * (slope_used.abs() as f64 + other_slope.abs() as f64) + gap
            };
            next_up_f32(val as f32)
        };

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a()[[j, i]];
                let ua = bounds.upper_a()[[j, i]];
                let l = pre_lower[i];
                let u = pre_upper[i];

                // Dual alpha (#3393): separate alpha values for lower and upper bound paths.
                // alpha_lower_i: used when la > 0 (maximizes lower bound)
                // alpha_upper_i: used when ua < 0 (minimizes upper bound)
                // Reference: auto_LiRPA/operators/relu.py selected_alpha[0] vs [1]
                let (alpha_lower_i, alpha_upper_i) = if l >= 0.0 {
                    (1.0, 1.0) // Always active
                } else if u <= 0.0 {
                    (0.0, 0.0) // Always inactive
                } else {
                    // Crossing: use provided alphas
                    let al = alpha[i];
                    let au = alpha_upper.map_or(al, |a| a[i]);
                    (al, au)
                };

                // For lower bound output: maximize lower
                // Guard: skip zero coefficients to avoid 0*inf NaN (#1739).
                // Directed rounding on slope products (#2786): next_down_f32 for lower_a,
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
                                    lower_b_err[j] += e as f64 * lambda_intercept[i].abs() as f64;
                                }
                            }
                        }
                    } else {
                        lower_nonfinite_rows[j] = true;
                    }
                    if track_gradients && is_crossing[i] {
                        // d(la * alpha * x)/d(alpha) at x = l_i gives la * l_i.
                        // The l_i < 0 factor is essential for correct sign.
                        // Reference: backward.rs AnalyticChain gradient. Fix: #3294
                        gradient_lower[i] += la * pre_lower[i];
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
                                lower_b_err[j] += e as f64 * lambda_intercept[i].abs() as f64;
                            }
                        }
                    } else {
                        lower_nonfinite_rows[j] = true;
                    }
                    new_lower_b_f64[j] += la as f64 * lambda_intercept[i] as f64;
                } else if track_err && la == 0.0 {
                    // Stored coefficient is exactly 0, but a_true ∈ [-e, e] may
                    // select either envelope: carry the full hull cover (no gap).
                    let e = in_lower_err.map_or(0.0, |m| m[[j, i]]);
                    if e != 0.0 {
                        new_lower_a_err[[j, i]] =
                            coeff_err_val(e, false, alpha_lower_i, lambda[i], 0.0);
                        lower_b_err[j] += e as f64 * lambda_intercept[i].abs() as f64;
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
                                upper_b_err[j] += e as f64 * lambda_intercept[i].abs() as f64;
                            }
                        }
                    } else {
                        upper_nonfinite_rows[j] = true;
                    }
                    new_upper_b_f64[j] += ua as f64 * lambda_intercept[i] as f64;
                } else if ua < 0.0 {
                    // Dual alpha (#3393): use alpha_upper for the ua < 0 case.
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
                                    upper_b_err[j] += e as f64 * lambda_intercept[i].abs() as f64;
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
                        upper_b_err[j] += e as f64 * lambda_intercept[i].abs() as f64;
                    }
                }
            }
        }

        // Fold the certified intercept (bias) error OUTWARD before the directed
        // cast: lower decreases, upper increases. No-op when !track_err (the
        // accumulators stay 0), preserving byte-identical output.
        if track_err {
            for j in 0..num_outputs {
                new_lower_b_f64[j] -= lower_b_err[j];
                new_upper_b_f64[j] += upper_b_err[j];
            }
        }

        // #3086: Zero affected rows and set bias to ±Inf for rows with non-finite
        // coefficient overflow. Only affected rows are degraded — all other rows
        // preserve full precision. Pattern matches common/mod.rs #3009.
        let lower_affected = lower_nonfinite_rows.iter().filter(|&&r| r).count();
        let upper_affected = upper_nonfinite_rows.iter().filter(|&&r| r).count();
        if lower_affected > 0 || upper_affected > 0 {
            debug!(
                "Alpha-CROWN ReLU backward: non-finite coeff×slope overflow in {}/{} lower rows, \
                 {}/{} upper rows — falling back to ±Inf bias for affected rows",
                lower_affected, num_outputs, upper_affected, num_outputs
            );
        }
        let mut new_lower_b = new_lower_b_f64.mapv(|v| next_down_f32(v as f32));
        let mut new_upper_b = new_upper_b_f64.mapv(|v| next_up_f32(v as f32));
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

        // CROWN backward NaN firewall (#2812): final safety net catches any
        // missed corruption beyond per-row tracking. When the incoming bounds
        // carry certified coefficient error, propagate the sign-stability composed
        // error (task #35); otherwise emit err-free bounds byte-identical to the
        // pre-#35 path.
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
            // Non-contiguous pre-activation (rare) skips the fold: the error
            // stays attached — still sound, just carries downstream at the
            // higher cost.
            if let (Some(pl), Some(pu)) = (pre_lower.as_slice(), pre_upper.as_slice()) {
                b.fold_coeff_err_into_bias(pl, pu);
            }
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
    /// directed rounding (`next_down_f32`/`next_up_f32`), same per-row f64 bias
    /// accumulation order, same certified coefficient-error carry, same non-finite
    /// row degrade, and the same constructors — so the certified bound is neither
    /// tighter nor looser, it is the SAME f32 bits. The bit-identity is asserted by
    /// the kernel parity test in `tests_soundness.rs`.
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
        e64 * (slope_used.abs() as f64 + other_slope.abs() as f64) + gap
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
    let mut new_lower_b_f64 = bounds.lower_b().mapv(|v| v as f64);
    let mut new_upper_b_f64 = bounds.upper_b().mapv(|v| v as f64);

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
                                lower_b_err[j] += e as f64 * lam_int.abs() as f64;
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
                            lower_b_err[j] += e as f64 * lam_int.abs() as f64;
                        }
                    }
                } else {
                    lower_nf[j] = true;
                }
                new_lower_b_f64[j] += la as f64 * lam_int as f64;
            } else if track_err && la == 0.0 {
                let e = in_lower_err.map_or(0.0, |m| m[[j, i]]);
                if e != 0.0 {
                    new_lower_a_err[[j, i]] =
                        relu_alpha_coeff_err_val(e, false, alpha_lower_i, lam, 0.0);
                    lower_b_err[j] += e as f64 * lam_int.abs() as f64;
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
                            upper_b_err[j] += e as f64 * lam_int.abs() as f64;
                        }
                    }
                } else {
                    upper_nf[j] = true;
                }
                new_upper_b_f64[j] += ua as f64 * lam_int as f64;
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
                                upper_b_err[j] += e as f64 * lam_int.abs() as f64;
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
                    upper_b_err[j] += e as f64 * lam_int.abs() as f64;
                }
            }
        }
    }

    if track_err {
        for j in 0..num_outputs {
            new_lower_b_f64[j] -= lower_b_err[j];
            new_upper_b_f64[j] += upper_b_err[j];
        }
    }

    let mut new_lower_b = new_lower_b_f64.mapv(|v| next_down_f32(v as f32));
    let mut new_upper_b = new_upper_b_f64.mapv(|v| next_up_f32(v as f32));
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
    let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
    let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);

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
            new_lower_b_f64[j] += lr.intercept_contrib;
            lower_nf[j] |= lr.nonfinite;

            let ur = compose::compose_upper(ua, relax);
            new_upper_a[[j, i]] = ur.new_coeff;
            new_upper_b_f64[j] += ur.intercept_contrib;
            upper_nf[j] |= ur.nonfinite;

            let slope_sum = (relax.lower_slope.abs() + relax.upper_slope.abs()) as f64;
            let int_sum = (relax.lower_intercept.abs() + relax.upper_intercept.abs()) as f64;

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
                        lower_b_err[j] += ea * int_cover;
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
                        upper_b_err[j] += ea * int_cover;
                    }
                }
            }
        }
        new_lower_b_f64[j] -= lower_b_err[j];
        new_upper_b_f64[j] += upper_b_err[j];
    }

    let mut new_lower_b = new_lower_b_f64.mapv(|x| next_down_f32(x as f32));
    let mut new_upper_b = new_upper_b_f64.mapv(|x| next_up_f32(x as f32));
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
    /// CROWN backward through ReLU in Patches mode with optimizable alpha.
    ///
    /// This is the Patches-mode counterpart of [`Self::propagate_linear_with_alpha`].
    /// Uses `alpha[i]` as the lower bound slope for crossing neurons instead of the
    /// heuristic value. Returns `(CrownBounds, gradient)` where `gradient[i]` =
    /// `d(sum of lower bounds)/d(alpha[i])`.
    ///
    /// Reference: alpha-beta-CROWN auto_LiRPA/operators/relu.py (Patches alpha backward)
    /// Part of #3293
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
    /// Design: designs/2026-02-28-patches-mode-wrapper-enum-design.md Phase 2
    /// Part of #2613
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

#[cfg(test)]
mod tests;
