// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ReLU backward propagation with α and β parameters.
//!
//! This is the production ReLU backward pass used by the β-CROWN verifier
//! during branch-and-bound. The α parameters control lower bound slopes
//! (optimized via gradient descent) and β parameters encode branching
//! constraints via Lagrangian relaxation.
//!
//! There is deliberately no cut-aware variant: the GCP-CROWN `arelu_cut` fold
//! was deleted along with the rest of the uncertified cut machinery. See
//! `BetaCrownConfig::cut_proof_authority_enabled()`.

use crate::layers::linear::bias::{add_f64_down, add_f64_up};
use ny_core::dd::next_up_f64;
use std::collections::HashMap;

use ndarray::Array2;
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::beta_crown::state::{BetaState, DomainAlphaState};
use crate::layers::activations::RELU_RELAX_MIN_WIDTH;
use crate::LinearBounds;

use super::super::BetaCrownVerifier;

impl BetaCrownVerifier {
    /// ReLU backward pass with both α and β parameters.
    // Justification: ReLU backward needs output bounds, pre-activation bounds,
    // branching constraints, beta/alpha state, and layer index — all from BaB context.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine) fn relu_backward_with_alpha_beta(
        &self,
        output_bounds: &LinearBounds,
        pre_bounds: &BoundedTensor,
        constraints: Option<&HashMap<usize, bool>>,
        beta_state: &BetaState,
        alpha_state: &DomainAlphaState,
        layer_idx: usize,
    ) -> Result<LinearBounds> {
        let pre_flat = pre_bounds.flatten();
        let num_neurons = pre_flat.len();
        let num_outputs = output_bounds.num_outputs();

        if output_bounds.num_inputs() != num_neurons {
            return Err(NyError::InternalError(format!(
                "ReLU backward (α,β) dimension mismatch at layer {}: output_bounds has {} inputs but layer has {} neurons",
                layer_idx,
                output_bounds.num_inputs(),
                num_neurons,
            )));
        }

        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        // f64 bias accumulators to prevent catastrophic cancellation.
        // Pattern matches legacy.rs and common/crown_dense.rs.
        let mut new_lower_b_f64 = output_bounds.lower_b().mapv(|x| x as f64);
        let mut new_upper_b_f64 = output_bounds.upper_b().mapv(|x| x as f64);
        let mut saw_nonfinite_coeff_product = false;

        for j in 0..num_neurons {
            let l = pre_flat.lower()[[j]];
            let u = pre_flat.upper()[[j]];
            let constraint = constraints.and_then(|c| c.get(&j).copied());

            let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
                if let Some(is_active) = constraint {
                    if is_active {
                        (1.0, 0.0, 1.0, 0.0)
                    } else {
                        (0.0, 0.0, 0.0, 0.0)
                    }
                } else if l.is_nan() || u.is_nan() {
                    // NaN bounds → fail closed to unbounded intercepts (sound).
                    (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY)
                } else if l >= 0.0 {
                    (1.0, 0.0, 1.0, 0.0)
                } else if u <= 0.0 {
                    (0.0, 0.0, 0.0, 0.0)
                } else if l.is_infinite() && u.is_infinite() {
                    // Both -Inf and +Inf: no finite affine upper envelope exists.
                    // Match relu_linear_relaxation() in relu/mod.rs.
                    (0.0, 0.0, 0.0, f32::INFINITY)
                } else if u.is_infinite() {
                    // Finite l < 0 < +Inf: chord limit → slope=1, intercept=-l.
                    // Match relu_linear_relaxation() in relu/mod.rs.
                    (1.0, 0.0, 1.0, -l)
                } else if l.is_infinite() {
                    // -Inf < 0 < finite u: tight upper envelope is constant y <= u.
                    // Match relu_linear_relaxation() in relu/mod.rs.
                    (0.0, 0.0, 0.0, u)
                } else {
                    // Clamp width to avoid division by zero when u ≈ l
                    let width = (u - l).max(RELU_RELAX_MIN_WIDTH);
                    let upper_slope_val = u / width;
                    let upper_intercept_val = -l * u / width;
                    let lower_slope_val = alpha_state.alpha(layer_idx, j);
                    (lower_slope_val, 0.0, upper_slope_val, upper_intercept_val)
                };

            // Match the shared compose policy in layers/common/compose.rs:
            // lower coefficients round toward -inf, upper coefficients toward +inf.
            for i in 0..num_outputs {
                let la_ij = output_bounds.lower_a()[[i, j]];
                let ua_ij = output_bounds.upper_a()[[i, j]];

                if la_ij > 0.0 {
                    let product = la_ij * lower_slope;
                    new_lower_a[[i, j]] = if product.is_finite() {
                        next_down_f32(product)
                    } else {
                        saw_nonfinite_coeff_product = true;
                        0.0
                    };
                    new_lower_b_f64[i] += la_ij as f64 * lower_intercept as f64;
                } else if la_ij < 0.0 {
                    let product = la_ij * upper_slope;
                    new_lower_a[[i, j]] = if product.is_finite() {
                        next_down_f32(product)
                    } else {
                        saw_nonfinite_coeff_product = true;
                        0.0
                    };
                    new_lower_b_f64[i] += la_ij as f64 * upper_intercept as f64;
                } else {
                    // Keep exact zero to avoid 0 * (+/-inf) -> NaN when NaN fallback is active.
                    new_lower_a[[i, j]] = 0.0;
                }

                if ua_ij > 0.0 {
                    let product = ua_ij * upper_slope;
                    new_upper_a[[i, j]] = if product.is_finite() {
                        next_up_f32(product)
                    } else {
                        saw_nonfinite_coeff_product = true;
                        0.0
                    };
                    new_upper_b_f64[i] += ua_ij as f64 * upper_intercept as f64;
                } else if ua_ij < 0.0 {
                    let product = ua_ij * lower_slope;
                    new_upper_a[[i, j]] = if product.is_finite() {
                        next_up_f32(product)
                    } else {
                        saw_nonfinite_coeff_product = true;
                        0.0
                    };
                    new_upper_b_f64[i] += ua_ij as f64 * lower_intercept as f64;
                } else {
                    // Keep exact zero to avoid 0 * (+/-inf) -> NaN when NaN fallback is active.
                    new_upper_a[[i, j]] = 0.0;
                }
            }

            if let Some(signed_beta) = beta_state.signed_beta(layer_idx, j) {
                // Skip non-finite beta to avoid poisoning the entire A-matrix.
                // Non-finite beta means the Lagrangian multiplier optimization produced
                // invalid output; skipping preserves valid pre-beta bounds (sound).
                if signed_beta.is_finite() {
                    // Magnitude of x_j over the pre-activation box, for the charge below.
                    // Same access pattern the slope computation above uses (:64-65).
                    let xmag =
                        (pre_flat.lower()[[j]].abs()).max(pre_flat.upper()[[j]].abs()) as f64;
                    let beta64 = signed_beta as f64;
                    for i in 0..num_outputs {
                        // The f32 mutation rounds. Measure the gap against the EXACT f64
                        // result (the f32->f64 widening is exact, so these references are
                        // the true intended values) and charge it into the bias.
                        let lo_before = new_lower_a[[i, j]];
                        let up_before = new_upper_a[[i, j]];
                        let lo_after = lo_before - signed_beta;
                        let up_after = up_before + signed_beta;
                        new_lower_a[[i, j]] = lo_after;
                        new_upper_a[[i, j]] = up_after;

                        // Sign-agnostic: |gap * x_j| <= gap * max(|l_j|, |u_j|), which is
                        // what makes a bias discharge sound for a SIGNED x_j where
                        // directing the coefficient would not be.
                        let lo_gap = ((lo_after as f64) - (lo_before as f64 - beta64)).abs();
                        let up_gap = ((up_after as f64) - (up_before as f64 + beta64)).abs();
                        // The charge itself must round UP, and its accumulation must be
                        // DIRECTED: a charge that rounds down shrinks the widening, and the
                        // `next_down_f32`/`next_up_f32` at publication (:181-182) covers only
                        // the final f64->f32 narrowing, not drift accumulated across j in f64.
                        if lo_gap != 0.0 {
                            new_lower_b_f64[i] =
                                add_f64_down(new_lower_b_f64[i], -next_up_f64(lo_gap * xmag));
                        }
                        if up_gap != 0.0 {
                            new_upper_b_f64[i] =
                                add_f64_up(new_upper_b_f64[i], next_up_f64(up_gap * xmag));
                        }
                    }
                } else {
                    tracing::warn!(
                        layer_idx,
                        neuron_idx = j,
                        signed_beta,
                        "Skipping non-finite beta contribution in relu_backward_with_alpha_beta"
                    );
                }
            }
        }

        if saw_nonfinite_coeff_product {
            return Ok(LinearBounds::conservative(num_outputs, num_neurons));
        }

        // Convert f64 bias accumulators back to f32 with directed rounding.
        let new_lower_b = new_lower_b_f64.mapv(|x| next_down_f32(x as f32));
        let new_upper_b = new_upper_b_f64.mapv(|x| next_up_f32(x as f32));

        LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
    }
}
