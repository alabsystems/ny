// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact-rational bound on the **deployed-network f32 rounding error** —
//! the margin that folds real-semantics certificates onto the deployed
//! network.
//!
//! Every certificate this crate emits proves a property of the network's
//! IDEAL real-valued semantics: f32 weights enter as their exact rational
//! values and all arithmetic is real (the same contract stated for the SMT
//! escalation path in `ny-groundtruth::escalate`). The *deployed* network,
//! however, executes in f32 round-to-nearest. This module computes an exact
//! rational `delta` such that for every input `x` in the box,
//!
//! ```text
//!   |fl32(net(x)) − net(x)| ≤ delta
//! ```
//!
//! where `fl32` is standard f32 inference (round-to-nearest-even per
//! operation) and `net` is the ideal real-valued semantics of the SAME
//! weights. Certifying `net(x) ≥ threshold + delta` (via
//! [`DeepReluProblem::certify`]'s `threshold` parameter) then proves
//! `fl32(net(x)) ≥ threshold` for the deployed network.
//!
//! # Method — classic layer-wise forward error analysis, computed exactly
//!
//! The analysis itself uses **no floating point**: every quantity is an
//! exact [`Rat`], so the analysis introduces no rounding concerns of its
//! own. With `u = 2⁻²⁴` (f32 unit roundoff), `η = 2⁻¹⁴⁹` (smallest positive
//! subnormal) and `γ_k = k·u / (1 − k·u)` (requiring `k·u < 1`):
//!
//! * **Magnitudes.** Per-neuron magnitude bounds `m_j` come from exact
//!   interval arithmetic over the input box ([`DeepReluProblem::preact_bounds`],
//!   plain exact Rat interval propagation): `m_j = max(|lo|, |hi|)` of the
//!   real-valued range (post-ReLU ranges clamp to `[0, max(0, hi)]`, so
//!   there `m_j = max(0, hi_j)`).
//! * **Affine layer** (`n` inputs, weight row `w_i`, bias `b_i`), where the
//!   incoming deployed activations carry accumulated error `e_j`:
//!   - dot+bias rounding error:
//!     `γ_{n+1} · (Σ_j |w_ij|·m_j + |b_i|) +
//!      2n·η/2·(1 + γ_{n+1})`
//!   - propagated input error: `(Σ_j |w_ij|·e_j) · (1 + γ_{n+1})`
//!   - `e_i^{new}` = the two terms summed. (The `(1 + γ_{n+1})` factor on the
//!     propagated term is exactly what absorbs the deployed magnitudes being
//!     `m_j + e_j` rather than `m_j` in the rounding term.)
//! * **ReLU:** `max(z, 0)` is exact in f32 and 1-Lipschitz, so the error
//!   vector passes through unchanged; magnitudes clamp to `[0, max(0, hi)]`.
//!
//! # Assumptions (documented soundness boundary)
//!
//! * **Per-operation relative error ≤ u** for round-to-nearest-even add and
//!   multiply (also valid for FMA, which incurs at most one rounding per
//!   term instead of two, so the bound only over-covers it).
//! * **Any summation order** is covered: a sequential sum of `k` terms
//!   passes each addend through at most `k` roundings, and every other
//!   association (pairwise/tree, height ≤ k) passes it through fewer, so
//!   `γ_{n+1}` bounds the dot+bias accumulation regardless of order.
//! * **Underflow:** an affine row performs `n` multiplies and at most `n`
//!   additions (including the bias). Each operation may contribute absolute
//!   error `η/2`; `2n·η/2·(1 + γ_{n+1})` covers all such errors and their
//!   amplification by later roundings.
//! * **The f32 weights ARE the deployed weights.** Every weight and bias is
//!   checked to be the exact rational lift of a finite f32 before it can
//!   contribute authority.
//! * **Inputs are exact:** the deployed network is fed the same f32 input
//!   values the real semantics is evaluated at (`e_j = 0` at the input
//!   layer). Input quantization error, if any, is out of scope here and
//!   must be added by the caller.
//! * **Overflow fails closed.** Each row's exact absolute-value envelope,
//!   propagated input error, relative rounding, and underflow slack must fit
//!   within finite f32 range. This also bounds every partial sum under any
//!   association because the envelope is the sum of absolute term bounds.

use num_bigint::BigInt;

use crate::crown_deep::{DeepCrownError, DeepReluProblem};
use crate::rational::{Rat, RatError};

/// Errors that can arise while computing a deployed-FP error margin.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum FpMarginError {
    /// The network failed validation or exact interval propagation.
    #[error(transparent)]
    Network(#[from] DeepCrownError),
    /// Exact arithmetic failure.
    #[error(transparent)]
    Rat(#[from] RatError),
    /// An affine layer has so many inputs that `(n+1)·2⁻²⁴ ≥ 1`, outside the
    /// `γ_k` model's domain (`k·u < 1`). Fail-closed; unreachable for any
    /// realistic network (needs width ≥ 2²⁴ − 1).
    #[error(
        "affine layer with {width} inputs is too wide for the f32 error model \
         (needs ({width}+1)·2^-24 < 1)"
    )]
    WidthTooLarge {
        /// Number of inputs of the offending affine row.
        width: usize,
    },
    /// A purported deployed weight or bias is not exactly representable as a
    /// finite f32, so there is no authenticated deployed parameter to compare
    /// with the ideal rational semantics.
    #[error("deployed parameter {value} is not the exact value of a finite f32")]
    NonF32Parameter {
        /// Canonical rational spelling of the rejected parameter.
        value: String,
    },
    /// A conservative magnitude envelope can reach beyond finite f32 range.
    #[error("deployed affine row may overflow f32 (magnitude envelope {magnitude})")]
    OverflowRisk {
        /// Canonical rational spelling of the conservative envelope.
        magnitude: String,
    },
}

/// Result alias for this module.
pub type FpMarginResult<T> = Result<T, FpMarginError>;

/// Per-neuron deployed-FP error bounds for a [`DeepReluProblem`].
///
/// All bounds are exact rationals: for every input `x` in the problem's box,
/// the deployed f32 value of the given neuron differs from its ideal
/// real-valued value by at most the stated bound (in absolute value).
#[derive(Debug, Clone)]
pub struct FpMargin {
    /// `activation_errors[L-1][j]` bounds `|fl32(a⁽ᴸ⁾ⱼ) − a⁽ᴸ⁾ⱼ|` — the
    /// post-ReLU activation error after hidden layer `L` (1-indexed).
    pub activation_errors: Vec<Vec<Rat>>,
    /// Bound on `|fl32(net(x)) − net(x)|` for the scalar read-out.
    pub output: Rat,
}

/// The f32 unit roundoff `u = 2⁻²⁴` (round-to-nearest half-ulp bound).
fn unit_roundoff() -> Result<Rat, RatError> {
    Rat::new(1, 1 << 24)
}

/// The smallest positive f32 subnormal `η = 2⁻¹⁴⁹`.
fn min_subnormal() -> Result<Rat, RatError> {
    Rat::from_bigints(BigInt::from(1), BigInt::from(1) << 149u32)
}

fn finite_f32_parameter(value: Rat) -> bool {
    let approximate = value.to_f64_approx();
    if !approximate.is_finite() {
        return false;
    }
    Rat::from_f32_exact(approximate as f32).is_some_and(|lifted| lifted == value)
}

fn require_finite_f32_parameter(value: Rat) -> FpMarginResult<()> {
    if finite_f32_parameter(value) {
        Ok(())
    } else {
        Err(FpMarginError::NonF32Parameter {
            value: value.to_clean_string()?,
        })
    }
}

/// `γ_{n+1} = (n+1)·u / (1 − (n+1)·u)`, failing closed when `(n+1)·u ≥ 1`.
fn gamma_row(n_inputs: usize, u: Rat) -> FpMarginResult<Rat> {
    let k = n_inputs
        .checked_add(1)
        .ok_or(FpMarginError::WidthTooLarge { width: n_inputs })?;
    let ku = Rat::from_int(k as i128).mul(u)?;
    if ku >= Rat::ONE {
        return Err(FpMarginError::WidthTooLarge { width: n_inputs });
    }
    Ok(ku.mul(Rat::ONE.sub(ku)?.inv()?)?)
}

/// Forward error bound for one affine row `w·a + b` evaluated in f32 on
/// deployed activations with real-magnitude bounds `mags` and accumulated
/// errors `errs`:
/// `γ_{n+1}·(Σ|w_j|·m_j + |b|) +
///  2n·η/2·(1+γ_{n+1}) + (Σ|w_j|·e_j)·(1 + γ_{n+1})`.
fn row_error(
    row: &[Rat],
    bias: Rat,
    mags: &[Rat],
    errs: &[Rat],
    u: Rat,
    eta: Rat,
) -> FpMarginResult<Rat> {
    let n = row.len();
    let gamma = gamma_row(n, u)?;
    require_finite_f32_parameter(bias)?;
    let mut mag_sum = bias.abs();
    let mut err_sum = Rat::ZERO;
    for ((w, m), e) in row.iter().zip(mags).zip(errs) {
        require_finite_f32_parameter(*w)?;
        let wa = w.abs();
        mag_sum = mag_sum.add(wa.mul(*m)?)?;
        err_sum = err_sum.add(wa.mul(*e)?)?;
    }
    // A separately-rounded dot+bias performs n multiplies and at most n
    // additions. Each absolute η/2 error may be amplified by the remaining
    // relative roundings, so multiply the aggregate by 1+γ. FMA performs
    // fewer roundings and is therefore covered by the same bound.
    let operation_count = n
        .checked_mul(2)
        .ok_or(FpMarginError::WidthTooLarge { width: n })?;
    let one_plus_gamma = Rat::ONE.add(gamma)?;
    let underflow = Rat::from_int(operation_count as i128)
        .mul(eta)?
        .mul(Rat::new(1, 2)?)?
        .mul(one_plus_gamma)?;
    let deployed_magnitude = mag_sum.add(err_sum)?.mul(one_plus_gamma)?.add(underflow)?;
    let max_finite = Rat::from_f32_exact(f32::MAX).ok_or(RatError::Poisoned)?;
    if deployed_magnitude > max_finite {
        return Err(FpMarginError::OverflowRisk {
            magnitude: deployed_magnitude.to_clean_string()?,
        });
    }
    let rounding = gamma.mul(mag_sum)?.add(underflow)?;
    let propagated = err_sum.mul(one_plus_gamma)?;
    Ok(rounding.add(propagated)?)
}

/// Compute the full per-neuron deployed-FP error bounds for `problem` over
/// its input box (see the module docs for the model and its assumptions).
///
/// # Errors
/// [`FpMarginError`] on an invalid network, exact-arithmetic failure, or an
/// affine layer too wide for the `γ_k` model.
pub fn deployed_fp_margin(problem: &DeepReluProblem) -> FpMarginResult<FpMargin> {
    // Exact interval propagation over the box (validates the network too):
    // real-valued pre-activation ranges, layer by layer.
    let preact = problem.preact_bounds()?;
    let u = unit_roundoff()?;
    let eta = min_subnormal()?;

    // Input magnitudes m_j = max(|lo|, |hi|); inputs are exact (e_j = 0).
    let mut mags: Vec<Rat> = Vec::new();
    for (lo, hi) in problem.input_lower.iter().zip(&problem.input_upper) {
        let (la, ha) = (lo.abs(), hi.abs());
        mags.push(if la >= ha { la } else { ha });
    }
    let mut errs: Vec<Rat> = vec![Rat::ZERO; mags.len()];

    let mut activation_errors: Vec<Vec<Rat>> = Vec::new();
    for ((w, b), z_hi) in problem
        .weights
        .iter()
        .zip(&problem.biases)
        .zip(&preact.upper)
    {
        // Affine layer: fresh error per neuron.
        let mut new_errs = Vec::new();
        for (row, bias) in w.iter().zip(b) {
            new_errs.push(row_error(row, *bias, &mags, &errs, u, eta)?);
        }
        // ReLU: exact in f32 and 1-Lipschitz — errors pass through unchanged;
        // real activation magnitudes clamp to [0, max(0, z_hi)].
        let mut new_mags = Vec::new();
        for hi in z_hi {
            new_mags.push(if hi.is_negative() { Rat::ZERO } else { *hi });
        }
        mags = new_mags;
        errs = new_errs.clone();
        activation_errors.push(new_errs);
    }

    let output = row_error(&problem.out_weight, problem.out_bias, &mags, &errs, u, eta)?;
    Ok(FpMargin {
        activation_errors,
        output,
    })
}

/// The scalar-output deployed-FP margin `delta` with
/// `|fl32(net(x)) − net(x)| ≤ delta` for every `x` in the box — the value to
/// fold into a certificate `threshold` so the real-semantics proof covers
/// the deployed network.
///
/// For a dominance / difference certificate (`f − g ≥ 0` with an exact
/// analytic ground-truth side `g`, as in `ny-groundtruth::certify_dominance`)
/// only `f` executes in f32, so the difference network's deployed error is
/// exactly this margin computed on `f`'s stack: certifying
/// `f(x) − g(x) ≥ delta` in real semantics proves
/// `fl32(f(x)) − g(x) ≥ 0` for the deployed network.
///
/// # Errors
/// See [`deployed_fp_margin`].
pub fn deployed_fp_output_margin(problem: &DeepReluProblem) -> FpMarginResult<Rat> {
    Ok(deployed_fp_margin(problem)?.output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i128, d: i128) -> Rat {
        Rat::new(n, d).expect("test rational")
    }

    /// 1 hidden layer, 2 inputs, hand-computed delta.
    ///
    /// Network: z = x0 + x1 + 1 (one hidden neuron), a = ReLU(z), y = 2a.
    /// Box [−1, 1]². Hand derivation with u = 2⁻²⁴, η = 2⁻¹⁴⁹:
    ///
    /// * input mags m = (1, 1); z ∈ [−1, 3] ⇒ activation mag m_a = 3.
    /// * hidden row (n = 2): γ₃ = 3u/(1−3u) = 3/(2²⁴−3) = 3/16777213;
    ///   `e_a = γ₃·(1·1 + 1·1 + |1|) + 4·η/2·(1+γ₃)`
    ///   `= 9/16777213 + 2η·16777216/16777213`.
    /// * read-out row (n = 1): γ₂ = 2u/(1−2u) = 2/(2²⁴−2) = 1/8388607;
    ///   `rounding = γ₂·(2·3 + 0) + 2·η/2·(1+γ₂)`
    ///   `= 6/8388607 + η·8388608/8388607`;
    ///   propagated = 2·e_a·(1 + γ₂) = 2·e_a·(8388608/8388607).
    /// * delta is the sum of that output rounding term and
    ///   `2·e_a·(8388608/8388607)`.
    #[test]
    fn test_fp_margin_one_layer_two_inputs_matches_hand_computation() {
        let problem = DeepReluProblem {
            weights: vec![vec![vec![Rat::ONE, Rat::ONE]]],
            biases: vec![vec![Rat::ONE]],
            out_weight: vec![r(2, 1)],
            out_bias: Rat::ZERO,
            input_lower: vec![r(-1, 1), r(-1, 1)],
            input_upper: vec![r(1, 1), r(1, 1)],
            alpha: None,
            interm_round: false,
        };
        let margin = deployed_fp_margin(&problem).expect("margin computes");

        // Independently hand-built expected value (see the doc comment).
        let eta = Rat::from_bigints(BigInt::from(1), BigInt::from(1) << 149u32).expect("eta");
        let e_a = r(9, 16_777_213)
            .add(
                r(2, 1)
                    .mul(eta)
                    .expect("2 eta")
                    .mul(r(16_777_216, 16_777_213))
                    .expect("amplified hidden underflow"),
            )
            .expect("hidden error");
        let expected = r(6, 8_388_607)
            .add(
                eta.mul(r(8_388_608, 8_388_607))
                    .expect("amplified output underflow"),
            )
            .expect("rounding")
            .add(
                r(2, 1)
                    .mul(e_a)
                    .expect("2 e_a")
                    .mul(r(8_388_608, 8_388_607))
                    .expect("1+gamma2"),
            )
            .expect("delta");

        assert_eq!(
            margin.activation_errors,
            vec![vec![e_a]],
            "hidden-layer error must match the hand computation"
        );
        assert_eq!(
            margin.output, expected,
            "output delta must match the hand computation exactly"
        );
        // Sanity: delta is tiny but strictly positive (≈ 1.79e-6).
        assert!(margin.output.is_positive(), "delta must be positive");
        assert!(
            margin.output < r(1, 100_000),
            "delta should be well below 1e-5 for this toy net, got {}/{}",
            margin.output.num(),
            margin.output.den()
        );
    }

    #[test]
    fn test_fp_margin_relu_passes_error_through_unchanged() {
        // Two hidden layers where the second is the 1×1 identity (w=1, b=0):
        // its row error must be γ₂·m + 2·η/2 + e·(1+γ₂) — strictly more than
        // e alone (the affine op itself rounds), and computable exactly.
        let problem = DeepReluProblem {
            weights: vec![vec![vec![Rat::ONE, Rat::ONE]], vec![vec![Rat::ONE]]],
            biases: vec![vec![Rat::ONE], vec![Rat::ZERO]],
            out_weight: vec![Rat::ONE],
            out_bias: Rat::ZERO,
            input_lower: vec![r(-1, 1), r(-1, 1)],
            input_upper: vec![r(1, 1), r(1, 1)],
            alpha: None,
            interm_round: false,
        };
        let margin = deployed_fp_margin(&problem).expect("margin computes");
        assert_eq!(margin.activation_errors.len(), 2);
        let e1 = margin.activation_errors[0][0];
        let e2 = margin.activation_errors[1][0];
        assert!(e2 > e1, "each affine layer strictly grows the error bound");
    }

    #[test]
    fn test_fp_margin_invalid_network_fails_closed() {
        let problem = DeepReluProblem {
            weights: vec![],
            biases: vec![],
            out_weight: vec![],
            out_bias: Rat::ZERO,
            input_lower: vec![],
            input_upper: vec![],
            alpha: None,
            interm_round: false,
        };
        assert!(matches!(
            deployed_fp_margin(&problem),
            Err(FpMarginError::Network(DeepCrownError::NoHiddenLayers))
        ));
    }

    #[test]
    fn test_fp_margin_rejects_non_f32_parameters_and_overflow() {
        let mut problem = DeepReluProblem {
            weights: vec![vec![vec![Rat::ONE]]],
            biases: vec![vec![Rat::ZERO]],
            out_weight: vec![Rat::ONE],
            out_bias: Rat::ZERO,
            input_lower: vec![Rat::ZERO],
            input_upper: vec![Rat::ONE],
            alpha: None,
            interm_round: false,
        };
        problem.weights[0][0][0] = r(1, 10);
        assert!(matches!(
            deployed_fp_margin(&problem),
            Err(FpMarginError::NonF32Parameter { .. })
        ));

        let max = Rat::from_f32_exact(f32::MAX).expect("f32::MAX lifts exactly");
        problem.weights[0][0][0] = max;
        problem.input_upper[0] = max;
        assert!(matches!(
            deployed_fp_margin(&problem),
            Err(FpMarginError::OverflowRisk { .. })
        ));
    }
}
