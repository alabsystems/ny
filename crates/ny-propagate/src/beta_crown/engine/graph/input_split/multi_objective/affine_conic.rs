// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified conic closure from already-computed CROWN source rows.
//!
//! If `l_i(x) <= o_i(x)` are sound affine lower relaxations and the unsafe
//! conjunction requires `o_0 <= t_0` and `o_1 <= t_1`, every finite
//! `lambda_0, lambda_1 >= 0` also requires
//!
//! ```text
//! lambda_0 l_0(x) + lambda_1 l_1(x)
//!     <= lambda_0 t_0 + lambda_1 t_1.
//! ```
//!
//! A strict lower bound in the opposite direction closes the domain. Combining
//! the source affine rows before concretization preserves input-coefficient
//! cancellation for adaptive weights. The authenticated verifier may run a
//! separate, selectively propagated unit-conic row as an independent
//! opportunity; this evaluator receives only the two-row source carrier.

use std::collections::HashSet;

use ny_core::{
    dd::{next_down_f64, next_up_f64},
    is_crown_coeff_safe,
};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::trace;

use crate::bounds::{certified_affine_sum_f32, LinearBounds, OutwardDirection};

/// This lane authenticates the four-input Cersyve property. Keep the helper
/// fail-open and absolutely bounded if a programmatic caller presents a much
/// wider lookalike.
const MAX_AFFINE_CONIC_INPUTS: usize = 4_096;
/// At most 32 central/error-shifted coefficient crossings, each expanded to
/// its nearest binary32 weight and two adjacent weights.
const MAX_CROSSING_PROPOSALS: usize = 32;

/// A sound evaluation of one authenticated conic combination over an input box.
#[derive(Clone, Copy, Debug)]
pub(super) struct ConicEvaluation {
    pub(super) lower_bound: f64,
    pub(super) threshold_upper: f64,
    pub(super) lhs_weight: f32,
    pub(super) rhs_weight: f32,
}

impl ConicEvaluation {
    #[inline]
    pub(super) fn verifies(self) -> bool {
        self.lower_bound > self.threshold_upper
    }

    /// Heuristic/telemetry value only. Proof authority is [`Self::verifies`].
    #[inline]
    pub(super) fn gap(self) -> f64 {
        self.lower_bound - self.threshold_upper
    }
}

#[inline]
fn push_candidate(
    candidates: &mut Vec<(f32, f32)>,
    seen: &mut HashSet<(u32, u32)>,
    lhs_weight: f32,
    rhs_weight: f32,
) {
    // Canonicalize signed zero before bitwise deduplication. Both spellings are
    // legitimate zero conic multipliers, but there is no value in evaluating
    // them twice.
    let lhs_weight = if lhs_weight == 0.0 { 0.0 } else { lhs_weight };
    let rhs_weight = if rhs_weight == 0.0 { 0.0 } else { rhs_weight };
    if !lhs_weight.is_finite()
        || !rhs_weight.is_finite()
        || lhs_weight < 0.0
        || rhs_weight < 0.0
        || (lhs_weight == 0.0 && rhs_weight == 0.0)
        || !seen.insert((lhs_weight.to_bits(), rhs_weight.to_bits()))
    {
        return;
    }
    candidates.push((lhs_weight, rhs_weight));
}

#[inline]
fn push_normalized_candidate(
    candidates: &mut Vec<(f32, f32)>,
    seen: &mut HashSet<(u32, u32)>,
    lhs_weight: f32,
) {
    if lhs_weight > 0.0 && lhs_weight < 1.0 && lhs_weight.is_finite() {
        // The subtraction may round, which is harmless: conic multipliers need
        // only be independently non-negative, not sum to one exactly.
        push_candidate(candidates, seen, lhs_weight, 1.0 - lhs_weight);
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_candidate(
    linear: &LinearBounds,
    in_lower: &[f32],
    in_upper: &[f32],
    thresholds: &[f32],
    lhs_weight: f32,
    rhs_weight: f32,
) -> Option<ConicEvaluation> {
    debug_assert_eq!(linear.num_outputs(), 2);
    debug_assert_eq!(linear.num_inputs(), in_lower.len());
    debug_assert_eq!(in_lower.len(), in_upper.len());
    debug_assert_eq!(thresholds.len(), 2);
    debug_assert!(lhs_weight.is_finite() && lhs_weight >= 0.0);
    debug_assert!(rhs_weight.is_finite() && rhs_weight >= 0.0);
    let lower_a = linear.lower_a();
    let lower_b = linear.lower_b();
    let lower_err = linear.lower_a_err();
    let mut lower_bound = certified_affine_sum_f32(
        0.0,
        [(lhs_weight, lower_b[0]), (rhs_weight, lower_b[1])],
        OutwardDirection::Lower,
    );
    if !lower_bound.is_finite() {
        return None;
    }

    for column in 0..linear.num_inputs() {
        let a0 = lower_a[[0, column]];
        let a1 = lower_a[[1, column]];
        let e0 = lower_err.map_or(0.0, |err| err[[0, column]]);
        let e1 = lower_err.map_or(0.0, |err| err[[1, column]]);

        // The true combined coefficient lies in [central_lo-E, central_hi+E].
        let central_lo = certified_affine_sum_f32(
            0.0,
            [(lhs_weight, a0), (rhs_weight, a1)],
            OutwardDirection::Lower,
        );
        let central_hi = certified_affine_sum_f32(
            0.0,
            [(lhs_weight, a0), (rhs_weight, a1)],
            OutwardDirection::Upper,
        );
        let error_hi = certified_affine_sum_f32(
            0.0,
            [(lhs_weight, e0), (rhs_weight, e1)],
            OutwardDirection::Upper,
        );
        let coeff_lo = next_down_f64(central_lo - error_hi);
        let coeff_hi = next_up_f64(central_hi + error_hi);
        if !coeff_lo.is_finite() || !coeff_hi.is_finite() || coeff_lo > coeff_hi {
            return None;
        }

        // Interval multiplication extrema occur at a corner. Round each
        // product down before taking the minimum.
        let products = [
            next_down_f64(coeff_lo * f64::from(in_lower[column])),
            next_down_f64(coeff_lo * f64::from(in_upper[column])),
            next_down_f64(coeff_hi * f64::from(in_lower[column])),
            next_down_f64(coeff_hi * f64::from(in_upper[column])),
        ];
        let term_lower = products.into_iter().fold(f64::INFINITY, f64::min);
        if !term_lower.is_finite() {
            return None;
        }
        lower_bound = next_down_f64(lower_bound + term_lower);
        if !lower_bound.is_finite() {
            return None;
        }
    }

    let threshold_upper = certified_affine_sum_f32(
        0.0,
        [(lhs_weight, thresholds[0]), (rhs_weight, thresholds[1])],
        OutwardDirection::Upper,
    );
    threshold_upper.is_finite().then_some(ConicEvaluation {
        lower_bound,
        threshold_upper,
        lhs_weight,
        rhs_weight,
    })
}

/// Evaluate a bounded family of sound non-negative conic combinations.
///
/// The equal-weight unit sum is always included. For tighter closure, the
/// evaluator also checks fixed interior weights and each stored-coefficient
/// zero crossing (plus its adjacent binary32 values). With exact affine rows,
/// the box minimum is concave piecewise-linear in the normalized weight and an
/// optimum occurs at an endpoint or such a crossing. The caller has already
/// tested both source-row endpoints through ordinary objective bounds;
/// coefficient uncertainty can shift the interior optimum, so this routine
/// makes no completeness claim. Every candidate it does evaluate remains
/// independently sound.
///
/// Returns `None` on every shape, layout, or numerical irregularity. The
/// coefficient-error carriers are part of the proof: row-wise errors are
/// combined outward before each coefficient is concretized, so cancellation of
/// stored central coefficients can never erase uncertainty.
pub(super) fn evaluate_affine_conic_closure(
    linear: &LinearBounds,
    input: &BoundedTensor,
    thresholds: &[f32],
) -> Option<ConicEvaluation> {
    if linear.num_outputs() != 2
        || thresholds.len() != 2
        || linear.num_inputs() != input.len()
        || linear.num_inputs() > MAX_AFFINE_CONIC_INPUTS
    {
        return None;
    }

    let in_lower = input.lower().as_slice()?;
    let in_upper = input.upper().as_slice()?;
    let lower_a = linear.lower_a();
    let lower_b = linear.lower_b();
    let lower_err = linear.lower_a_err();
    if in_lower.len() != linear.num_inputs()
        || in_upper.len() != linear.num_inputs()
        || lower_a.nrows() != 2
        || lower_b.len() != 2
        || lower_err.is_some_and(|err| err.shape() != lower_a.shape())
        || lower_b.iter().any(|value| !value.is_finite())
        || thresholds.iter().any(|value| !value.is_finite())
        // Match the ordinary CROWN concretization trust firewall. Finite but
        // near-overflow coefficients are deliberately not verdict-authorized;
        // cancellation against another such row must not restore authority.
        || lower_a
            .iter()
            .any(|value| !is_crown_coeff_safe(*value))
        || lower_err.is_some_and(|err| {
            err.iter()
                .any(|value| *value < 0.0 || !is_crown_coeff_safe(*value))
        })
        || in_lower
            .iter()
            .zip(in_upper)
            .any(|(&lower, &upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
    {
        return None;
    }

    let mut candidates = Vec::with_capacity(6 + 3 * MAX_CROSSING_PROPOSALS);
    let mut seen = HashSet::with_capacity(candidates.capacity());
    // Endpoints can be tighter here than ordinary concretization on a
    // one-sided box because interval multiplication retains the coefficient
    // sign/error interaction instead of paying `err * max_abs(input)`.
    push_candidate(&mut candidates, &mut seen, 1.0, 0.0);
    push_candidate(&mut candidates, &mut seen, 0.0, 1.0);
    push_candidate(&mut candidates, &mut seen, 1.0, 1.0);
    for weight in [0.25, 0.5, 0.75] {
        push_normalized_candidate(&mut candidates, &mut seen, weight);
    }

    // For c(w)=w*a0+(1-w)*a1, a coefficient changes sign at
    // w=-a1/(a0-a1). Include the central row and both coefficient-error
    // interval boundaries; adjacent f32 weights cover either side of a rounded
    // root. Rank once by the kink's potential slope change over this box,
    // retain a hard-bounded portfolio of distinct binary32 centers, then
    // evaluate it in linear time with respect to input width.
    let mut crossing_proposals = Vec::with_capacity(3 * linear.num_inputs());
    for column in 0..linear.num_inputs() {
        let a0 = f64::from(lower_a[[0, column]]);
        let a1 = f64::from(lower_a[[1, column]]);
        let e0 = f64::from(lower_err.map_or(0.0, |err| err[[0, column]]));
        let e1 = f64::from(lower_err.map_or(0.0, |err| err[[1, column]]));
        for (candidate_a0, candidate_a1) in [(a0, a1), (a0 - e0, a1 - e1), (a0 + e0, a1 + e1)] {
            let denominator = candidate_a0 - candidate_a1;
            if denominator == 0.0 || !denominator.is_finite() {
                continue;
            }
            let crossing = -candidate_a1 / denominator;
            if crossing > 0.0 && crossing < 1.0 && crossing.is_finite() {
                let width = f64::from(in_upper[column]) - f64::from(in_lower[column]);
                crossing_proposals.push((denominator.abs() * width, column, crossing));
            }
        }
    }
    crossing_proposals.sort_unstable_by(|lhs, rhs| {
        rhs.0
            .total_cmp(&lhs.0)
            .then_with(|| lhs.1.cmp(&rhs.1))
            .then_with(|| lhs.2.total_cmp(&rhs.2))
    });
    let available = crossing_proposals.len();
    let mut crossing_centers = HashSet::with_capacity(MAX_CROSSING_PROPOSALS);
    for (_, _, crossing) in crossing_proposals {
        let center = crossing as f32;
        // Zero error makes all three boundary proposals identical. Deduplicate
        // before applying the portfolio limit so repeats cannot crowd out
        // lower-ranked but genuinely different kinks.
        if !crossing_centers.insert(center.to_bits()) {
            continue;
        }
        for weight in [next_down_f32(center), center, next_up_f32(center)] {
            push_normalized_candidate(&mut candidates, &mut seen, weight);
        }
        if crossing_centers.len() == MAX_CROSSING_PROPOSALS {
            if available > crossing_centers.len() {
                trace!(
                    available,
                    retained = crossing_centers.len(),
                    "[multi-obj] truncating affine conic crossing portfolio"
                );
            }
            break;
        }
    }

    let mut best_nonverifying = None;
    for (lhs_weight, rhs_weight) in candidates {
        if let Some(evaluation) = evaluate_candidate(
            linear, in_lower, in_upper, thresholds, lhs_weight, rhs_weight,
        ) {
            // A direct comparison is the proof predicate. Return immediately so
            // a rounded telemetry gap can never displace a verifying candidate.
            if evaluation.verifies() {
                return Some(evaluation);
            }
            if best_nonverifying
                .is_none_or(|best: ConicEvaluation| evaluation.gap().total_cmp(&best.gap()).is_gt())
            {
                best_nonverifying = Some(evaluation);
            }
        }
    }
    best_nonverifying
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, Array1, Array2};

    use super::*;

    fn box_2d(lower: [f32; 2], upper: [f32; 2]) -> BoundedTensor {
        BoundedTensor::new(arr1(&lower).into_dyn(), arr1(&upper).into_dyn()).unwrap()
    }

    #[test]
    fn cancellation_closes_without_materializing_another_crown_row() {
        let linear = LinearBounds::new(
            arr2(&[[100_000_000.0, 1.0], [-100_000_000.0, 1.0]]),
            arr1(&[0.0, 0.0]),
            arr2(&[[100_000_000.0, 1.0], [-100_000_000.0, 1.0]]),
            arr1(&[0.0, 0.0]),
        )
        .unwrap();
        let input = box_2d([-1.0, 1.0], [1.0, 1.0]);

        let evaluation = evaluate_affine_conic_closure(&linear, &input, &[0.0, -0.0]).unwrap();
        assert!(evaluation.verifies(), "evaluation={evaluation:?}");
        assert!(evaluation.lower_bound > 0.99, "evaluation={evaluation:?}");
    }

    #[test]
    fn adaptive_crossing_closes_when_unit_sum_and_source_rows_do_not() {
        // l0=x+0.2<=0 requires x<=-0.2; l1=-2x+0.2<=0 requires x>=0.1.
        // Neither source row nor their unit sum closes [-1,1], but weights
        // (2/3,1/3) cancel x and leave the strictly-positive constant 0.2.
        let linear = LinearBounds::new(
            arr2(&[[1.0], [-2.0]]),
            arr1(&[0.2, 0.2]),
            arr2(&[[1.0], [-2.0]]),
            arr1(&[0.2, 0.2]),
        )
        .unwrap();
        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

        let unit = evaluate_candidate(&linear, &[-1.0], &[1.0], &[0.0, -0.0], 1.0, 1.0).unwrap();
        assert!(!unit.verifies(), "unit={unit:?}");
        let evaluation = evaluate_affine_conic_closure(&linear, &input, &[0.0, -0.0]).unwrap();
        assert!(evaluation.verifies(), "evaluation={evaluation:?}");
        assert!(evaluation.lhs_weight > evaluation.rhs_weight);
    }

    #[test]
    fn coefficient_error_survives_stored_coefficient_cancellation() {
        let mut linear = LinearBounds::new(
            arr2(&[[1.0, 0.0], [-1.0, 0.0]]),
            arr1(&[0.000_1, 0.0]),
            arr2(&[[1.0, 0.0], [-1.0, 0.0]]),
            arr1(&[0.000_1, 0.0]),
        )
        .unwrap();
        linear.set_coeff_err(arr2(&[[0.001, 0.0], [0.0, 0.0]]), Array2::zeros((2, 2)));
        let input = box_2d([-1.0, 0.0], [1.0, 0.0]);

        let evaluation = evaluate_affine_conic_closure(&linear, &input, &[0.0, -0.0]).unwrap();
        assert!(!evaluation.verifies(), "evaluation={evaluation:?}");
        assert!(evaluation.lower_bound < 0.0, "evaluation={evaluation:?}");
    }

    #[test]
    fn nonpositive_sum_fails_open() {
        let linear = LinearBounds::new(
            arr2(&[[1.0], [-1.0]]),
            arr1(&[0.0, 0.0]),
            arr2(&[[1.0], [-1.0]]),
            arr1(&[0.0, 0.0]),
        )
        .unwrap();
        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

        let evaluation = evaluate_affine_conic_closure(&linear, &input, &[0.0, -0.0]).unwrap();
        assert!(!evaluation.verifies(), "evaluation={evaluation:?}");
    }

    #[test]
    fn wrong_row_or_threshold_shape_declines() {
        let linear = LinearBounds::identity(2);
        let input = box_2d([-1.0, -1.0], [1.0, 1.0]);
        assert!(evaluate_affine_conic_closure(&linear, &input, &[0.0]).is_none());

        let one_row = LinearBounds::new(
            arr2(&[[1.0, 0.0]]),
            arr1(&[0.0]),
            arr2(&[[1.0, 0.0]]),
            arr1(&[0.0]),
        )
        .unwrap();
        assert!(evaluate_affine_conic_closure(&one_row, &input, &[0.0, -0.0]).is_none());
    }

    #[test]
    fn near_overflow_coefficients_and_errors_never_regain_authority_by_cancellation() {
        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
        let unsafe_coefficients = LinearBounds::new(
            arr2(&[[1.0e10], [-1.0e10]]),
            arr1(&[1.0, 1.0]),
            arr2(&[[1.0e10], [-1.0e10]]),
            arr1(&[1.0, 1.0]),
        )
        .unwrap();
        assert!(
            evaluate_affine_conic_closure(&unsafe_coefficients, &input, &[0.0, -0.0]).is_none()
        );

        let mut unsafe_error = LinearBounds::new(
            arr2(&[[1.0], [-1.0]]),
            arr1(&[1.0, 1.0]),
            arr2(&[[1.0], [-1.0]]),
            arr1(&[1.0, 1.0]),
        )
        .unwrap();
        unsafe_error.set_coeff_err(arr2(&[[1.0e10], [0.0]]), Array2::zeros((2, 1)));
        assert!(evaluate_affine_conic_closure(&unsafe_error, &input, &[0.0, -0.0]).is_none());
    }

    #[test]
    fn nonstandard_input_layout_and_excessive_width_decline() {
        let lower = arr2(&[[-1.0, -1.0], [-1.0, -1.0]])
            .reversed_axes()
            .into_dyn();
        let upper = arr2(&[[1.0, 1.0], [1.0, 1.0]]).reversed_axes().into_dyn();
        assert!(lower.as_slice().is_none());
        let nonstandard = BoundedTensor::new(lower, upper).unwrap();
        let four_inputs = LinearBounds::new(
            Array2::zeros((2, 4)),
            arr1(&[0.0, 0.0]),
            Array2::zeros((2, 4)),
            arr1(&[0.0, 0.0]),
        )
        .unwrap();
        assert!(evaluate_affine_conic_closure(&four_inputs, &nonstandard, &[0.0, -0.0]).is_none());

        let width = MAX_AFFINE_CONIC_INPUTS + 1;
        let wide_input = BoundedTensor::new(
            Array1::from_elem(width, -1.0).into_dyn(),
            Array1::from_elem(width, 1.0).into_dyn(),
        )
        .unwrap();
        let wide_rows = LinearBounds::new(
            Array2::zeros((2, width)),
            arr1(&[0.0, 0.0]),
            Array2::zeros((2, width)),
            arr1(&[0.0, 0.0]),
        )
        .unwrap();
        assert!(evaluate_affine_conic_closure(&wide_rows, &wide_input, &[0.0, -0.0]).is_none());
    }
}
