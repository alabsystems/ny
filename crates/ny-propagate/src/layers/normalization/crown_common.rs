// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared scalar CROWN backward propagation for normalization layers.
//!
//! This module contains the sampling-based CROWN linearization algorithm
//! that is generic over any [`NormLayer`] implementor. Previously this
//! code was duplicated 4x across LayerNorm, RmsNorm, InstanceNorm1d,
//! and AdaIN1d crown_scalar.rs files (~870 lines of identical logic).
//!
//! The batched CROWN counterpart lives in [`super::crown_batched_common`].
//!
//! # Algorithm overview
//!
//! 1. Mode gating (Sound → error, Cut → identity, Sampling → proceed)
//! 2. Flatten pre-activation bounds to 1D
//! 3. Compute center-point eval/Jacobian
//! 4. Sampling-based error estimation (3-level grid + axis-aligned + hash-random)
//! 5. f64 backward propagation with per-row non-finite guard
//! 6. Directed rounding on bias
//!
//! Reference: designs/2026-02-27-normalization-trait-dedup.md

use ndarray::{Array1, Array2};
use ny_core::{is_crown_coeff_safe_f64, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::warn;

use super::layer_norm::types::LayerNormCrownMode;
use super::trait_norm::NormLayer;
use crate::LinearBounds;

/// Gate the CROWN mode and return early for Sound/Cut.
///
/// - `IbpValidated`: return `Ok(None)` (caller routes to its sound decomposed
///   primitive-chain CROWN, never to [`sampling_crown_scalar`])
/// - `Sound`: return `SoundnessRefusal` error
/// - `Cut`: return `Ok(Some(bounds.clone()))` (identity relaxation)
/// - `Sampling`: return `Ok(None)` (caller proceeds with sampling)
pub(crate) fn gate_crown_mode<L: NormLayer>(
    layer: &L,
    bounds: &LinearBounds,
) -> Result<Option<LinearBounds>> {
    match layer.crown_mode() {
        LayerNormCrownMode::IbpValidated => {
            // Every caller intercepts IbpValidated and routes it to its sound
            // decomposed primitive-chain CROWN (#3775); only Sampling mode
            // reaches the shared sampling linearization below.
            Ok(None)
        }
        LayerNormCrownMode::Sound => Err(NyError::SoundnessRefusal(format!(
            "{} CROWN linearization uses heuristic sampling (not provably sound). \
             For sound verification, use IBP or cut CROWN at {} boundaries. \
             To proceed with sampling anyway, use the sampling mode.",
            layer.layer_name(),
            layer.layer_name(),
        ))),
        LayerNormCrownMode::Cut => {
            // Identity relaxation: pass bounds through unchanged (sound but loses correlations)
            Ok(Some(bounds.clone()))
        }
        LayerNormCrownMode::Sampling => {
            warn!(
                "{} using sampling-based CROWN linearization (not provably sound)",
                layer.layer_name()
            );
            Ok(None)
        }
    }
}

/// Flatten pre-activation bounds to 1D arrays (lower, upper).
pub(crate) fn flatten_preactivation(
    pre_activation: &BoundedTensor,
) -> Result<(Array1<f32>, Array1<f32>)> {
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
    Ok((pre_lower, pre_upper))
}

/// Sampling-based CROWN scalar linearization, generic over normalization type.
///
/// This is the core CROWN backward algorithm for normalization layers.
/// The caller handles mode gating (including any layer-specific early returns
/// like LayerNorm MeanOnly) and shape validation before calling this function.
///
/// # Parameters
///
/// - `layer`: the normalization layer (provides eval/jacobian)
/// - `bounds`: incoming CROWN linear bounds to propagate backward
/// - `pre_lower`, `pre_upper`: flattened 1D pre-activation bounds
///
/// # Panics
///
/// The caller must ensure `pre_lower.len() == pre_upper.len() == bounds.num_inputs()`.
pub(crate) fn sampling_crown_scalar<L: NormLayer>(
    layer: &L,
    bounds: &LinearBounds,
    pre_lower: &Array1<f32>,
    pre_upper: &Array1<f32>,
) -> Result<LinearBounds> {
    let num_neurons = pre_lower.len();
    let num_outputs = bounds.num_outputs();

    // Non-finite pre-activation guard (#3259, same pattern as #2591).
    // When any dimension has infinite or NaN bounds, sampling-based
    // linearization fails (center = NaN from (-inf + inf)/2).
    // Normalization is NOT the identity function — returning bounds.clone()
    // (identity passthrough) would be unsound.
    //
    // Return trivially sound constant bounds: A = 0, bias = [-inf, +inf].
    // This means "the output could be anything" — sound but contributes no
    // tightening. Matches the per-row non-finite guard in backward_propagate_f64
    // and the LogSoftmax constant-bounds fallback for #2591.
    let has_non_finite =
        pre_lower.iter().any(|&v| !v.is_finite()) || pre_upper.iter().any(|&v| !v.is_finite());
    if has_non_finite {
        return LinearBounds::new_or_conservative(
            Array2::zeros((num_outputs, num_neurons)),
            Array1::from_elem(num_outputs, f32::NEG_INFINITY),
            Array2::zeros((num_outputs, num_neurons)),
            Array1::from_elem(num_outputs, f32::INFINITY),
        );
    }

    // Compute center point and evaluate.
    // Use l + (u - l) / 2 instead of (l + u) / 2 to avoid overflow when
    // l and u are both large finite values (l + u could exceed f32::MAX).
    let x_center: Array1<f32> = pre_lower
        .iter()
        .zip(pre_upper.iter())
        .map(|(&l, &u)| l + (u - l) / 2.0)
        .collect();

    let y_center = layer.eval(&x_center)?;
    let jacobian = layer.jacobian(&x_center)?;

    // Guard: if Jacobian or eval output contains non-finite entries (Inf from
    // ny/std overflow when std is tiny), return NumericalInstability so the
    // caller falls back to IBP. (#2901)
    if jacobian.iter().any(|v| !v.is_finite()) || y_center.iter().any(|v| !v.is_finite()) {
        return Err(NyError::NumericalInstability(format!(
            "{} CROWN: non-finite Jacobian or eval output (ny/std overflow)",
            layer.layer_name(),
        )));
    }

    // Linear approximation: y ≈ J @ x + (y_c - J @ x_c)
    // where b_approx = y_c - J @ x_c
    let jx_center = jacobian.dot(&x_center);
    let b_approx: Array1<f32> = &y_center - &jx_center;

    // --- Sampling-based error estimation ---
    //
    // For small dimensions (n ≤ 6), sample a {lower, center, upper}^n grid
    // (3^n points). The normalization denominator depends on ALL inputs
    // simultaneously, so the worst nonlinear residual can occur at interior
    // points where some coordinates are at their center value (minimizing
    // the denominator). Corner-only sampling (2^n) misses these interior
    // extrema. (#3103)
    let (max_error_above, max_error_below) = estimate_sampling_error(
        layer,
        num_neurons,
        pre_lower,
        pre_upper,
        &x_center,
        &jacobian,
        &b_approx,
    )?;

    // Apply safety margin and minimum floor
    let (max_error_above, max_error_below) = apply_safety_margin(max_error_above, max_error_below);

    // Backward propagation with f64 accumulation (#1745, #2169).
    // The Jacobian has mixed signs (rows sum to 0 for LayerNorm), so
    // O(num_neurons) terms with cancellation make f32 accumulation unsound.
    // Directed rounding on final f32 cast (#1992, #2164).
    backward_propagate_f64(
        num_outputs,
        num_neurons,
        bounds,
        &jacobian,
        &b_approx,
        &max_error_above,
        &max_error_below,
    )
}

/// Sampling-based error estimation: 3-level grid + axis-aligned + hash-random.
fn estimate_sampling_error<L: NormLayer>(
    layer: &L,
    num_neurons: usize,
    pre_lower: &Array1<f32>,
    pre_upper: &Array1<f32>,
    x_center: &Array1<f32>,
    jacobian: &Array2<f32>,
    b_approx: &Array1<f32>,
) -> Result<(Array1<f32>, Array1<f32>)> {
    let max_grid_dims = 6; // 3^6 = 729 grid points max
    let num_grid = if num_neurons <= max_grid_dims {
        3_usize.pow(num_neurons as u32)
    } else {
        0
    };
    // 3-level grid captures center-value extrema; axis-aligned only for large n
    let num_axis_aligned = if num_grid > 0 { 0 } else { num_neurons * 2 };
    let num_random = 50;
    let num_samples = (num_grid + num_axis_aligned + num_random).max(50);

    let mut max_error_above: Array1<f32> = Array1::zeros(num_neurons);
    let mut max_error_below: Array1<f32> = Array1::zeros(num_neurons);

    let mut x_sample = x_center.clone();
    for sample_idx in 0..num_samples {
        if sample_idx < num_grid {
            // 3-level grid: {lower, center, upper}^n
            let mut grid_idx = sample_idx;
            for i in 0..num_neurons {
                let level = grid_idx % 3;
                grid_idx /= 3;
                x_sample[i] = match level {
                    0 => pre_lower[i],
                    1 => pre_lower[i] + (pre_upper[i] - pre_lower[i]) * 0.5,
                    _ => pre_upper[i],
                };
            }
        } else if sample_idx < num_grid + num_axis_aligned {
            // Axis-aligned corner sampling (fallback for large n)
            x_sample.assign(x_center);
            let offset = sample_idx - num_grid;
            let dim = offset / 2;
            if dim < num_neurons {
                x_sample[dim] = if offset % 2 == 0 {
                    pre_lower[dim]
                } else {
                    pre_upper[dim]
                };
            }
        } else {
            // Hash-based pseudo-random sampling
            for i in 0..num_neurons {
                let t = ((sample_idx as u32).wrapping_mul(2654435761_u32) ^ (i as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                x_sample[i] = pre_lower[i] + (pre_upper[i] - pre_lower[i]) * t;
            }
        }

        let y_actual = layer.eval(&x_sample)?;
        // Use f64 accumulation for linear approximation to avoid catastrophic
        // cancellation — normalization Jacobians have mixed signs and rows that
        // sum to near zero. Matches the f64 pattern in backward_propagate_f64.
        let y_approx: Array1<f32> = {
            let mut approx = Array1::<f64>::zeros(num_neurons);
            for i in 0..num_neurons {
                let mut sum = b_approx[i] as f64;
                for k in 0..num_neurons {
                    sum += jacobian[[i, k]] as f64 * x_sample[k] as f64;
                }
                approx[i] = sum;
            }
            approx.mapv(|v| v as f32)
        };

        for i in 0..num_neurons {
            let error = y_actual[i] - y_approx[i];
            if error > max_error_above[i] {
                max_error_above[i] = error;
            }
            if -error > max_error_below[i] {
                max_error_below[i] = -error;
            }
        }
    }

    Ok((max_error_above, max_error_below))
}

/// Apply safety margin (50% extra) and minimum floor (1e-6).
fn apply_safety_margin(
    mut max_error_above: Array1<f32>,
    mut max_error_below: Array1<f32>,
) -> (Array1<f32>, Array1<f32>) {
    let safety_factor = 1.5;
    let min_margin = 1e-6_f32;
    for i in 0..max_error_above.len() {
        max_error_above[i] *= safety_factor;
        max_error_below[i] *= safety_factor;
        if max_error_above[i] < min_margin {
            max_error_above[i] = min_margin;
        }
        if max_error_below[i] < min_margin {
            max_error_below[i] = min_margin;
        }
    }
    (max_error_above, max_error_below)
}

/// Backward propagation through normalization using linear relaxation.
///
/// Both weight and bias accumulation use f64 to prevent catastrophic
/// cancellation (#1745, #2169). Per-row non-finite guard (#3128, #3027)
/// widens affected rows to conservative bounds. Directed rounding on
/// final f32 bias cast (#1992, #2164).
fn backward_propagate_f64(
    num_outputs: usize,
    num_neurons: usize,
    bounds: &LinearBounds,
    jacobian: &Array2<f32>,
    b_approx: &Array1<f32>,
    max_error_above: &Array1<f32>,
    max_error_below: &Array1<f32>,
) -> Result<LinearBounds> {
    let mut new_lower_a_f64 = Array2::<f64>::zeros((num_outputs, num_neurons));
    let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
    let mut new_upper_a_f64 = Array2::<f64>::zeros((num_outputs, num_neurons));
    let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);

    for j in 0..num_outputs {
        for i in 0..num_neurons {
            let la = bounds.lower_a()[[j, i]];
            let ua = bounds.upper_a()[[j, i]];

            // For lower bound output: need lower bound on each y_i when coeff positive
            // Guard: skip zero coefficients to avoid 0*inf NaN (#1739).
            if la > 0.0 {
                let la_f64 = la as f64;
                for k in 0..num_neurons {
                    new_lower_a_f64[[j, k]] += la_f64 * jacobian[[i, k]] as f64;
                }
                new_lower_b_f64[j] += la_f64 * (b_approx[i] as f64 - max_error_below[i] as f64);
            } else if la < 0.0 {
                let la_f64 = la as f64;
                for k in 0..num_neurons {
                    new_lower_a_f64[[j, k]] += la_f64 * jacobian[[i, k]] as f64;
                }
                new_lower_b_f64[j] += la_f64 * (b_approx[i] as f64 + max_error_above[i] as f64);
            }

            // For upper bound output
            if ua > 0.0 {
                let ua_f64 = ua as f64;
                for k in 0..num_neurons {
                    new_upper_a_f64[[j, k]] += ua_f64 * jacobian[[i, k]] as f64;
                }
                new_upper_b_f64[j] += ua_f64 * (b_approx[i] as f64 + max_error_above[i] as f64);
            } else if ua < 0.0 {
                let ua_f64 = ua as f64;
                for k in 0..num_neurons {
                    new_upper_a_f64[[j, k]] += ua_f64 * jacobian[[i, k]] as f64;
                }
                new_upper_b_f64[j] += ua_f64 * (b_approx[i] as f64 - max_error_below[i] as f64);
            }
        }

        // Per-row unsafe coefficient guard (#3128, #3027, #3228): when Inf or
        // near-overflow coefficients from compose() produce Inf*0.0 = NaN or
        // Inf*nonzero = ±Inf in Jacobian accumulation, widen the affected row
        // to conservative bounds rather than poisoning the entire matrix via
        // new_or_conservative's global fallback. Uses is_crown_coeff_safe_f64()
        // (finite + magnitude ≤ CROWN_COEFF_MAX) to also catch near-overflow
        // before cascade to f32.
        let lower_row_nonfinite =
            (0..num_neurons).any(|k| !is_crown_coeff_safe_f64(new_lower_a_f64[[j, k]]));
        let upper_row_nonfinite =
            (0..num_neurons).any(|k| !is_crown_coeff_safe_f64(new_upper_a_f64[[j, k]]));
        if lower_row_nonfinite {
            for k in 0..num_neurons {
                new_lower_a_f64[[j, k]] = 0.0;
            }
            new_lower_b_f64[j] = f64::NEG_INFINITY;
        }
        if upper_row_nonfinite {
            for k in 0..num_neurons {
                new_upper_a_f64[[j, k]] = 0.0;
            }
            new_upper_b_f64[j] = f64::INFINITY;
        }
    }

    // A-matrix: standard f64→f32 rounding (round-to-nearest), matching
    // alpha-beta-CROWN. Directed rounding on A is not unconditionally sound
    // because the sign of the coefficient determines which direction is
    // conservative during concretization (#2208).
    LinearBounds::new_or_conservative(
        new_lower_a_f64.mapv(|x| x as f32),
        new_lower_b_f64.mapv(|x| next_down_f32(x as f32)),
        new_upper_a_f64.mapv(|x| x as f32),
        new_upper_b_f64.mapv(|x| next_up_f32(x as f32)),
    )
}
