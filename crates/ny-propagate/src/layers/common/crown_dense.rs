// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dense CROWN backward for element-wise activation functions.
//!
//! Operates on `Array2` A-matrices (output_neurons × input_neurons).

use ndarray::Array2;
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::compose;
use crate::layers::activations::LinearRelaxation;
use crate::LinearBounds;

/// Helper function for CROWN backward propagation through element-wise activation functions.
///
/// Delegates to [`crown_elementwise_backward_indexed`] with a no-op neuron index.
pub(crate) fn crown_elementwise_backward<F>(
    bounds: &LinearBounds,
    pre_activation: &BoundedTensor,
    relaxation_fn: F,
) -> Result<LinearBounds>
where
    F: Fn(f32, f32) -> LinearRelaxation,
{
    crown_elementwise_backward_indexed(bounds, pre_activation, |l, u, _i| relaxation_fn(l, u))
}

/// Indexed variant of CROWN backward for element-wise activations.
///
/// Like [`crown_elementwise_backward`], but the relaxation function receives the
/// neuron index `i` (column of the weight matrix). This supports layers where
/// the relaxation depends on a per-neuron parameter (e.g., PReLU's per-channel slopes).
///
/// # Arguments
/// * `bounds` - Incoming linear bounds from layers above
/// * `pre_activation` - Pre-activation bounds for this layer's inputs
/// * `relaxation_fn` - `(l, u, neuron_idx) -> LinearRelaxation`
pub fn crown_elementwise_backward_indexed<F>(
    bounds: &LinearBounds,
    pre_activation: &BoundedTensor,
    relaxation_fn: F,
) -> Result<LinearBounds>
where
    F: Fn(f32, f32, usize) -> LinearRelaxation,
{
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

    let num_outputs = bounds.num_outputs();

    // Precompute per-neuron relaxation parameters
    let pre_lower_slice = pre_lower
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous pre_lower array".into()))?;
    let pre_upper_slice = pre_upper
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous pre_upper array".into()))?;
    let relaxations =
        compose::precompute_relaxations(pre_lower_slice, pre_upper_slice, &relaxation_fn);

    // Backward propagation: compose the linear relaxation.
    // Bias accumulation uses f64 to prevent catastrophic cancellation (#1745).
    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);

    // Certified coefficient-error propagation through the relaxation
    // (#vnncomp-aw-soundness). For input coefficient `a` with certified error
    // `err_a`, the composed coefficient `a·slope(sign a)` has certified error
    // `err_a·(|lower_slope| + |upper_slope|) + directed-rounding-gap` (the
    // slope-sum term covers a possible sign-flip of `a` selecting the other
    // envelope), and the intercept contribution picks up
    // `err_a·(|lower_intercept| + |upper_intercept|)` folded into the bias
    // widening. Validated at 0 violations / 6M trials.
    //
    // SOUNDNESS (#vnncomp-aw-activation): the composed coefficient is the
    // directed-rounded f32 of the f32 product `a·slope`, which is NOT exactly
    // the true real coefficient `a_f64·slope_f64`. The `next_down_f32`/
    // `next_up_f32` in `compose_*` only makes the stored coefficient sound for
    // concretizing AT THIS layer's input box; once this activation's output
    // bounds are composed by a FURTHER backward layer, that layer reads the
    // stored f32 coefficient and (absent a certified error) treats it as exact,
    // dropping the directed-rounding gap and UNDER-counting the true error =
    // false-proof risk. So we ALWAYS carry the per-coefficient
    // directed-rounding gap `|a_f64·slope_f64 − stored_f32|` in the error
    // matrices, regardless of whether the incoming bounds carry error, and add
    // the propagated incoming-error term only when present.
    let in_lower_err = bounds.lower_a_err();
    let in_upper_err = bounds.upper_a_err();
    let mut new_lower_a_err = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_upper_a_err = Array2::<f32>::zeros((num_outputs, num_neurons));
    // Per-row certified intercept (bias) error, subtracted from lower / added to upper.
    let mut lower_b_err = vec![0.0f64; num_outputs];
    let mut upper_b_err = vec![0.0f64; num_outputs];

    // Track which output rows have non-finite coefficients after coeff × slope (#3009).
    let mut lower_nonfinite_rows = vec![false; num_outputs];
    let mut upper_nonfinite_rows = vec![false; num_outputs];

    for j in 0..num_outputs {
        for i in 0..num_neurons {
            let la = bounds.lower_a()[[j, i]];
            let ua = bounds.upper_a()[[j, i]];

            let lr = compose::compose_lower(la, &relaxations[i]);
            new_lower_a[[j, i]] = lr.new_coeff;
            new_lower_b_f64[j] += lr.intercept_contrib;
            lower_nonfinite_rows[j] |= lr.nonfinite;

            let ur = compose::compose_upper(ua, &relaxations[i]);
            new_upper_a[[j, i]] = ur.new_coeff;
            new_upper_b_f64[j] += ur.intercept_contrib;
            upper_nonfinite_rows[j] |= ur.nonfinite;

            // ALWAYS certify the per-coefficient error (#vnncomp-aw-activation).
            // Two independent contributions:
            //   1. directed-rounding gap = |a_f64·slope_used_f64 − stored_f32|,
            //      the exact distance between the f32 composed coefficient and
            //      the real product. Present even with NO incoming error.
            //   2. incoming-error term, with a SIGN-STABILITY refinement
            //      (#cgan-conv-err-compose, the fix named in
            //      docs/CGAN_BOUND_QUALITY_ROOT_CAUSE.md): the true incoming
            //      coefficient lies in `[a−e, a+e]`.
            //      - When `|a| > e`, the true coefficient has the SAME SIGN as
            //        `a`, so the envelope line selected by `compose_*` (chosen by
            //        sign) is the valid sound substitute for the true coefficient
            //        too: `err_out = e·|slope_chosen| + gap` and the bias picks up
            //        only `e·|intercept_chosen|` (the same line's intercept scaled
            //        by the coefficient uncertainty). No doubling: for ReLU the
            //        chosen slope is ≤ 1, so carried error can only SHRINK here —
            //        this kills the 2^L growth across L stable-identity ReLU
            //        layers that previously made the carried error reach
            //        intermediate-box scale on deep conv stacks.
            //      - When `|a| ≤ e`, the sign may flip and select the other
            //        envelope: keep the unconditional two-line cover
            //        `e·(|lower_slope|+|upper_slope|)` + `e·(|i_l|+|i_u|)` into
            //        the bias. This branch now also covers `a == 0` with `e > 0`
            //        (previously the incoming error was silently DROPPED for
            //        exact-zero coefficients — an unsound hole in the
            //        false-proof direction).
            let relax = &relaxations[i];
            let slope_sum = (relax.lower_slope.abs() + relax.upper_slope.abs()) as f64;
            let int_sum = (relax.lower_intercept.abs() + relax.upper_intercept.abs()) as f64;

            // Lower direction.
            {
                let gap = if la != 0.0 {
                    (la as f64 * lr_slope(la, relax) - lr.new_coeff as f64).abs()
                } else {
                    0.0
                };
                let ea = in_lower_err.map_or(0.0, |e| e[[j, i]] as f64);
                if gap != 0.0 || ea != 0.0 {
                    let (slope_cover, int_cover) = if (la as f64).abs() > ea {
                        // Sign-stable: the chosen line covers the true coefficient.
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

            // Upper direction.
            {
                let gap = if ua != 0.0 {
                    (ua as f64 * ur_slope(ua, relax) - ur.new_coeff as f64).abs()
                } else {
                    0.0
                };
                let ea = in_upper_err.map_or(0.0, |e| e[[j, i]] as f64);
                if gap != 0.0 || ea != 0.0 {
                    let (slope_cover, int_cover) = if (ua as f64).abs() > ea {
                        // Sign-stable: the chosen line covers the true coefficient.
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
        // Fold the certified intercept error into the bias accumulators BEFORE
        // the directed cast: lower decreases, upper increases.
        new_lower_b_f64[j] -= lower_b_err[j];
        new_upper_b_f64[j] += upper_b_err[j];
    }

    // #3009: Non-finite row fallback — zero A-row, set bias to ±Inf (sound but maximally loose).
    let lower_affected = lower_nonfinite_rows.iter().filter(|&&r| r).count();
    let upper_affected = upper_nonfinite_rows.iter().filter(|&&r| r).count();
    compose::log_nonfinite_fallback("Activation", lower_affected, upper_affected, num_outputs);
    // Outward-round the f64 bias into f32 — preserving EXACT ZERO. An
    // unconditional `next_down_f32(0.0)` manufactures a -1e-45 bias on rows
    // whose true bias is exactly 0.0: a needless 1-ulp loosening, and a
    // violation of the stacked-seed invariant that zero rows stay exactly
    // zero through every admitted step (#cgan-stacked-backward injection
    // guard refuses the whole pass on a nonzero find). `x == 0.0` casts
    // exactly, so no widening is required there.
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
        if lower_nonfinite_rows[j] {
            for i in 0..num_neurons {
                new_lower_a[[j, i]] = 0.0;
                new_lower_a_err[[j, i]] = 0.0;
            }
            new_lower_b[j] = f32::NEG_INFINITY;
        }
        if upper_nonfinite_rows[j] {
            for i in 0..num_neurons {
                new_upper_a[[j, i]] = 0.0;
                new_upper_a_err[[j, i]] = 0.0;
            }
            new_upper_b[j] = f32::INFINITY;
        }
    }

    // CROWN backward NaN firewall (#2812): falls back to conservative bounds instead
    // of aborting the verification chain when relaxation produces NaN/Inf coefficients.
    // Non-finite rows already zeroed above, but missed corruption is caught here.
    //
    // We ALWAYS attach the certified coefficient error (#vnncomp-aw-activation):
    // the composed f32 coefficient differs from the true real product by the
    // directed-rounding gap, which a downstream backward layer must not drop.
    LinearBounds::new_or_conservative_with_err(
        new_lower_a,
        new_lower_b,
        new_upper_a,
        new_upper_b,
        new_lower_a_err,
        new_upper_a_err,
    )
}

/// The lower-direction slope actually used by [`compose::compose_lower`] for a
/// coefficient of the given sign (lower_slope for `a>0`, upper_slope for `a<0`).
#[inline]
fn lr_slope(a: f32, relax: &LinearRelaxation) -> f64 {
    if a > 0.0 {
        relax.lower_slope as f64
    } else if a < 0.0 {
        relax.upper_slope as f64
    } else {
        0.0
    }
}

/// The upper-direction slope actually used by [`compose::compose_upper`].
#[inline]
fn ur_slope(a: f32, relax: &LinearRelaxation) -> f64 {
    if a > 0.0 {
        relax.upper_slope as f64
    } else if a < 0.0 {
        relax.lower_slope as f64
    } else {
        0.0
    }
}
