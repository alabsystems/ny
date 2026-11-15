// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2};
use ny_core::{
    dd::{next_down_f64, next_up_f64},
    f64_to_f32_down, f64_to_f32_up, Result,
};
use ny_tensor::{next_down_f32, next_up_f32};

use crate::bounds::nan_propagating_max;
use crate::LinearBounds;

/// Small epsilon for numerical stability.
pub(super) const SOFTMAX_EPSILON: f32 = 1e-12;
pub(super) const SOFTMAX_SANITIZE_MARGIN: f32 = 1e-6;

// =============================================================================
// Internal array-based helpers (used by layers)
// =============================================================================

pub(super) fn logsumexp_1d(values: &Array1<f32>) -> f32 {
    let max_val = values
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, nan_propagating_max);
    if !max_val.is_finite() {
        return max_val;
    }
    // f64 accumulation for sum of exponentials.
    let max_f64 = max_val as f64;
    let sum_exp: f64 = values.iter().map(|&v| (v as f64 - max_f64).exp()).sum();
    (max_f64 + sum_exp.ln()) as f32
}

pub(super) fn softmax_1d(values: &Array1<f32>) -> Array1<f32> {
    let max_val = values
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, nan_propagating_max);
    // f64 accumulation for exp and sum to prevent precision loss.
    let max_f64 = max_val as f64;
    let exp_vals_f64: Array1<f64> = values.mapv(|v| (v as f64 - max_f64).exp());
    let sum_exp: f64 = exp_vals_f64.sum();
    exp_vals_f64.mapv(|v| (v / sum_exp) as f32)
}

pub(super) fn sanitize_softmax_unit_bounds(mut lower: f32, mut upper: f32) -> (f32, f32) {
    // Softmax outputs are always in [0, 1]. When we detect numerical issues, prefer a
    // conservative widening rather than propagating NaN/inf.
    if !lower.is_finite() || !(0.0..=1.0).contains(&lower) {
        lower = 0.0;
    } else {
        lower = (lower - SOFTMAX_SANITIZE_MARGIN).max(0.0);
    }

    if !upper.is_finite() || upper < 0.0 {
        upper = 1.0;
    } else {
        upper = (upper + SOFTMAX_SANITIZE_MARGIN).min(1.0);
    }

    if lower > upper {
        (0.0, 1.0)
    } else {
        (lower, upper)
    }
}

// =============================================================================
// CROWN affine backward composition (shared by softmax + logsoftmax)
// =============================================================================

/// Compose upstream CROWN linear bounds with per-neuron affine relaxations.
///
/// Given upstream bounds `bounds` (output <- neuron) and per-neuron affine
/// relaxations `lower_a/b`, `upper_a/b` (neuron <- input), produces the
/// composed bounds (output <- input) using the standard CROWN backward rule:
///
///   new_lower[out, k] = sum_i { la_i * (la_i > 0 ? lower_a : upper_a)[i, k] }
///   new_upper[out, k] = sum_i { ua_i * (ua_i > 0 ? upper_a : lower_a)[i, k] }
///
/// # Certificate (#vnncomp-aw-soundness)
///
/// Every summand `la * relax_a[i, k]` is a product of two binary32 values, so
/// it is EXACT in binary64 (24 + 24 = 48 <= 53 significand bits, and the f32
/// exponent range cannot leave binary64's normal range). The ONLY arithmetic
/// error in the fold is therefore the binary64 `+=` chain itself, bounded by
/// Higham Thm 3.1 as `gamma_n^f64 * S` with `S = SUM |term|` — the MAGNITUDE
/// sum, which is what makes the bound survive the catastrophic cancellation
/// this function exists to fight (softmax/logsoftmax Jacobian rows have
/// mixed-sign terms that sum near zero). `n` is the number of
/// `+=` performed on that accumulator; it is identical for every column `k` of
/// a row, because the inner `k` loop runs for EVERY `i` whose incoming
/// coefficient is nonzero.
///
/// The A-matrix is then stored with round-to-nearest (a DIRECTED cast on `A`
/// is not unconditionally sound — the sign of the coefficient's contribution
/// over the box is not known here), and the resulting deviation is
/// published as the certified per-coefficient error
///   `err[out, k] = |f64_sum - stored_f32| + gamma_n^f64 * S`
/// via [`LinearBounds::new_or_conservative_with_err`], exactly as the MaxPool
/// dense CROWN backward does (`layers/pooling/max.rs`) and as the linear
/// backward's `aw_f64_with_abssum` does.
///
/// HISTORY — THE COMMENT WAS THE DEFECT. This site previously published
/// through `LinearBounds::new_or_conservative`, whose `lower_a_err`/
/// `upper_a_err` are `None`, and `None` is documented on the struct field as
/// the STRONGEST possible claim: "exact (error 0)". The old comment justified
/// that with "A-matrix uses nearest rounding", which is a correct
/// statement about the ROUNDING MODE and no justification at all for the
/// EXACTNESS CLAIM: nearest rounding is precisely why an error certificate is
/// required. Both the f64->f32 cast (up to 0.5 ulp per coefficient) and the
/// f64 accumulation went uncharged, so `concretize_sound` omitted the
/// `sum_j err[i,j] * max(|in_l_j|, |in_u_j|)` penalty it owed and the
/// concretized bound could be TIGHTER THAN TRUE.
///
/// SCOPE OF THIS CERTIFICATE. `true_coeff` here is the exact-real composition
/// of the binary32 inputs this function is GIVEN. It does NOT cover the error
/// in the relaxation coefficients themselves: `lower_a`/`upper_a` arrive as
/// bare `f32` with no carrier to attach a construction error to, and their
/// exp/sum/divide chains live in the callers (`linear/sound.rs`,
/// `logsoftmax/crown.rs`), where the intercepts `lower_b`/`upper_b` are
/// derived from the STORED f32 slopes (so the relaxation is self-consistent at
/// the expansion point by construction). A certificate for the relaxation
/// construction has to be produced there and threaded in; it cannot be
/// invented at this seam.
///
/// # Bias
///
/// The bias fold is the same shape of defect and is charged the same way. It
/// is a mixed-sign binary64 `+=` chain SEEDED from `bounds.lower_b()` /
/// `bounds.upper_b()`, so Higham's `SUM |addend|` must include that seed. A
/// bare directed binary32 cast of the result (`next_down_f32(x as f32)`)
/// covers only ~1 ulp OF THE RESULT and is therefore insufficient exactly when
/// the chain cancels — `Sum|addends| / |result|` unbounded is the regime a
/// softmax Jacobian row lives in. The charge is subtracted (lower) / added
/// (upper) in binary64 with directed rounding BEFORE the directed cast, and
/// the original `next_down_f32`/`next_up_f32` step is retained on top so the
/// published bias is never tighter than the previous implementation's for any
/// input.
///
/// Previously duplicated in `logsoftmax/crown.rs` and `linear/sound.rs`.
pub(super) fn apply_affine_bounds_f64(
    bounds: &LinearBounds,
    lower_a: &Array2<f32>,
    lower_b: &Array1<f32>,
    upper_a: &Array2<f32>,
    upper_b: &Array1<f32>,
) -> Result<LinearBounds> {
    let num_outputs = bounds.num_outputs();
    let num_neurons = lower_b.len();

    let mut new_lower_a_f64 = Array2::<f64>::zeros((num_outputs, num_neurons));
    let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
    let mut new_upper_a_f64 = Array2::<f64>::zeros((num_outputs, num_neurons));
    let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);

    // Higham accumulators: `*_s_a[[out, k]] = SUM |term|` for that coefficient,
    // `*_s_b[out] = |seed| + SUM |term|` for that bias (the seed is an addend of
    // the same chain), and `*_terms[out]` the number of `+=` on that row.
    let mut lower_s_a = Array2::<f64>::zeros((num_outputs, num_neurons));
    let mut upper_s_a = Array2::<f64>::zeros((num_outputs, num_neurons));
    let mut lower_s_b: Vec<f64> = new_lower_b_f64.iter().map(|&x: &f64| x.abs()).collect();
    let mut upper_s_b: Vec<f64> = new_upper_b_f64.iter().map(|&x: &f64| x.abs()).collect();
    let mut lower_terms = vec![0usize; num_outputs];
    let mut upper_terms = vec![0usize; num_outputs];

    for out_idx in 0..num_outputs {
        for i in 0..num_neurons {
            let la = bounds.lower_a()[[out_idx, i]];
            let ua = bounds.upper_a()[[out_idx, i]];

            // Guard: skip zero coefficients to avoid 0*inf NaN.
            if la > 0.0 {
                let la_f64 = la as f64;
                for k in 0..num_neurons {
                    let term = la_f64 * lower_a[[i, k]] as f64;
                    new_lower_a_f64[[out_idx, k]] += term;
                    lower_s_a[[out_idx, k]] += term.abs();
                }
                let term_b = la_f64 * lower_b[i] as f64;
                new_lower_b_f64[out_idx] += term_b;
                lower_s_b[out_idx] += term_b.abs();
                lower_terms[out_idx] += 1;
            } else if la < 0.0 {
                let la_f64 = la as f64;
                for k in 0..num_neurons {
                    let term = la_f64 * upper_a[[i, k]] as f64;
                    new_lower_a_f64[[out_idx, k]] += term;
                    lower_s_a[[out_idx, k]] += term.abs();
                }
                let term_b = la_f64 * upper_b[i] as f64;
                new_lower_b_f64[out_idx] += term_b;
                lower_s_b[out_idx] += term_b.abs();
                lower_terms[out_idx] += 1;
            }

            if ua > 0.0 {
                let ua_f64 = ua as f64;
                for k in 0..num_neurons {
                    let term = ua_f64 * upper_a[[i, k]] as f64;
                    new_upper_a_f64[[out_idx, k]] += term;
                    upper_s_a[[out_idx, k]] += term.abs();
                }
                let term_b = ua_f64 * upper_b[i] as f64;
                new_upper_b_f64[out_idx] += term_b;
                upper_s_b[out_idx] += term_b.abs();
                upper_terms[out_idx] += 1;
            } else if ua < 0.0 {
                let ua_f64 = ua as f64;
                for k in 0..num_neurons {
                    let term = ua_f64 * lower_a[[i, k]] as f64;
                    new_upper_a_f64[[out_idx, k]] += term;
                    upper_s_a[[out_idx, k]] += term.abs();
                }
                let term_b = ua_f64 * lower_b[i] as f64;
                new_upper_b_f64[out_idx] += term_b;
                upper_s_b[out_idx] += term_b.abs();
                upper_terms[out_idx] += 1;
            }
        }
    }

    // A-matrix: round-to-nearest store PLUS its certificate (see doc comment).
    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_lower_a_err = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_upper_a_err = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_lower_b = Array1::<f32>::zeros(num_outputs);
    let mut new_upper_b = Array1::<f32>::zeros(num_outputs);

    for out_idx in 0..num_outputs {
        let lower_gamma = crate::layers::linear::crown_single_gamma_n_f64(lower_terms[out_idx]);
        let upper_gamma = crate::layers::linear::crown_single_gamma_n_f64(upper_terms[out_idx]);

        for k in 0..num_neurons {
            let lower_value = new_lower_a_f64[[out_idx, k]];
            let lower_stored = lower_value as f32;
            new_lower_a[[out_idx, k]] = lower_stored;
            let lower_cast = (lower_value - lower_stored as f64).abs();
            let lower_sum = lower_s_a[[out_idx, k]];
            // `S == 0` means every addend was exactly zero, so the exact-real
            // sum and the stored value are both zero: the error really is zero
            // and claiming it keeps untouched columns out of the row-degrade
            // path. Any other case pays the full certificate, rounded up.
            new_lower_a_err[[out_idx, k]] = if lower_sum == 0.0 && lower_cast == 0.0 {
                0.0
            } else {
                next_up_f32((lower_cast + lower_gamma * lower_sum) as f32)
            };

            let upper_value = new_upper_a_f64[[out_idx, k]];
            let upper_stored = upper_value as f32;
            new_upper_a[[out_idx, k]] = upper_stored;
            let upper_cast = (upper_value - upper_stored as f64).abs();
            let upper_sum = upper_s_a[[out_idx, k]];
            new_upper_a_err[[out_idx, k]] = if upper_sum == 0.0 && upper_cast == 0.0 {
                0.0
            } else {
                next_up_f32((upper_cast + upper_gamma * upper_sum) as f32)
            };
        }

        // Bias: charge the f64 chain's own accumulation residual OUTWARD in f64
        // (the seed counts as an addend, hence `terms + 1`), then take the
        // directed f32 cast. `f64_to_f32_down(x) <= x as f32` and
        // `f64_to_f32_up(x) >= x as f32` in every branch, and the charge is
        // non-negative, so each published bias is <= (resp. >=) what the
        // previous `next_down_f32(x as f32)` / `next_up_f32(x as f32)` line
        // produced: this move is OUTWARD or equal for every input.
        let lower_charge =
            crate::layers::linear::crown_single_gamma_n_f64(lower_terms[out_idx].saturating_add(1))
                * lower_s_b[out_idx];
        let upper_charge =
            crate::layers::linear::crown_single_gamma_n_f64(upper_terms[out_idx].saturating_add(1))
                * upper_s_b[out_idx];
        new_lower_b[out_idx] = next_down_f32(f64_to_f32_down(next_down_f64(
            new_lower_b_f64[out_idx] - lower_charge,
        )));
        new_upper_b[out_idx] = next_up_f32(f64_to_f32_up(next_up_f64(
            new_upper_b_f64[out_idx] + upper_charge,
        )));
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

// =============================================================================
// Counterexamples for the CROWN composition certificate (#vnncomp-aw-soundness)
// =============================================================================
//
// These exercise the WIRING (`apply_affine_bounds_f64` end to end), not a
// kernel in isolation: each one drives the real composition and inspects what
// the function PUBLISHES. All three fail on the pre-fix implementation, which
// published through `LinearBounds::new_or_conservative` (coefficient error
// `None` = "exact") and cast the cancelling f64 bias chain with a single
// directed f32 step.
#[cfg(test)]
mod composition_certificate_tests {
    use super::*;
    use ndarray::{arr1, arr2, Array2};

    /// The composed A-matrix is stored with round-to-nearest, so it can differ
    /// from the exact-real composition by far more than the published bias
    /// margin covers. Publishing it as EXACT is the defect.
    ///
    /// Column 0 accumulates `1 + 2^-30 + 2^-30 = 1 + 2^-29`, which is exact in
    /// binary64 and rounds to exactly `1.0` in binary32 (`ulp(1) = 2^-23`). The
    /// stored coefficient is therefore `2^-29` away from the value it claims to
    /// be, and with `lower_a_err = None` `concretize_sound` charges nothing for
    /// it.
    #[ntest::timeout(10000)]
    #[test]
    fn apply_affine_bounds_f64_certifies_the_coefficient_cast_deviation() {
        let bounds = LinearBounds::new(
            arr2(&[[1.0f32, 1.0, 1.0]]),
            arr1(&[0.0f32]),
            arr2(&[[1.0f32, 1.0, 1.0]]),
            arr1(&[0.0f32]),
        )
        .expect("valid upstream bounds");

        let tiny = 2f32.powi(-30);
        let relax_a = arr2(&[[1.0f32, 0.0, 0.0], [tiny, 0.0, 0.0], [tiny, 0.0, 0.0]]);
        let relax_b = arr1(&[0.0f32, 0.0, 0.0]);

        let out = apply_affine_bounds_f64(&bounds, &relax_a, &relax_b, &relax_a, &relax_b)
            .expect("composition succeeds");

        assert_eq!(
            out.lower_a()[[0, 0]],
            1.0,
            "the stored coefficient is still the nearest f32"
        );
        let deviation = 2f32.powi(-29);

        assert!(
            out.has_coeff_err(),
            "the composition must publish a coefficient certificate, not claim exactness"
        );
        let lower_err = out
            .lower_a_err()
            .expect("lower coefficients must carry a certificate");
        let upper_err = out
            .upper_a_err()
            .expect("upper coefficients must carry a certificate");
        assert!(
            lower_err[[0, 0]] >= deviation,
            "lower certificate {} must cover the {} cast deviation",
            lower_err[[0, 0]],
            deviation
        );
        assert!(
            upper_err[[0, 0]] >= deviation,
            "upper certificate {} must cover the {} cast deviation",
            upper_err[[0, 0]],
            deviation
        );
    }

    /// The bias fold is a MIXED-SIGN binary64 `+=` chain, so a directed f32 cast
    /// of the RESULT does not cover its accumulation residual once the chain
    /// cancels.
    ///
    /// Upper chain: seed `1.0`, then `+2^-53`, `+2^-53`, `-1.0`. Each add of
    /// `2^-53` to `1.0` is an exact binary64 TIE that rounds to even and is
    /// discarded, so the computed result is exactly `0.0` while the exact-real
    /// sum is `2^-52`. `next_up_f32(0.0)` publishes `1.4e-45`, which is BELOW
    /// the true bias: the upper bound is tighter than truth.
    #[ntest::timeout(10000)]
    #[test]
    fn apply_affine_bounds_f64_upper_bias_encloses_a_cancelling_f64_chain() {
        let bounds = LinearBounds::new(
            arr2(&[[0.0f32, 0.0, 0.0]]),
            arr1(&[0.0f32]),
            arr2(&[[1.0f32, 1.0, 1.0]]),
            arr1(&[1.0f32]),
        )
        .expect("valid upstream bounds");

        let half_ulp = 2f32.powi(-53);
        let relax_a = Array2::<f32>::zeros((3, 3));
        let relax_b = arr1(&[half_ulp, half_ulp, -1.0f32]);

        let out = apply_affine_bounds_f64(&bounds, &relax_a, &relax_b, &relax_a, &relax_b)
            .expect("composition succeeds");

        let exact = 2f32.powi(-52);
        assert!(
            out.upper_b()[0] >= exact,
            "published upper bias {} must enclose the exact-real chain value {}",
            out.upper_b()[0],
            exact
        );
    }

    /// Mirror of the upper case on the lower side: seed `-1.0`, then `-2^-53`,
    /// `-2^-53`, `+1.0`. Computed `0.0`, exact-real `-2^-52`, and
    /// `next_down_f32(0.0) = -1.4e-45` sits ABOVE the true bias.
    #[ntest::timeout(10000)]
    #[test]
    fn apply_affine_bounds_f64_lower_bias_encloses_a_cancelling_f64_chain() {
        let bounds = LinearBounds::new(
            arr2(&[[1.0f32, 1.0, 1.0]]),
            arr1(&[-1.0f32]),
            arr2(&[[0.0f32, 0.0, 0.0]]),
            arr1(&[0.0f32]),
        )
        .expect("valid upstream bounds");

        let half_ulp = 2f32.powi(-53);
        let relax_a = Array2::<f32>::zeros((3, 3));
        let relax_b = arr1(&[-half_ulp, -half_ulp, 1.0f32]);

        let out = apply_affine_bounds_f64(&bounds, &relax_a, &relax_b, &relax_a, &relax_b)
            .expect("composition succeeds");

        let exact = -2f32.powi(-52);
        assert!(
            out.lower_b()[0] <= exact,
            "published lower bias {} must enclose the exact-real chain value {}",
            out.lower_b()[0],
            exact
        );
    }
}

#[cfg(test)]
mod tests;
