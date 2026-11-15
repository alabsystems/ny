// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified reductions of exact binary32 products.

use ny_core::dd::{dd_fma, gamma_n_dd, next_down_f64, next_up_f64, Dd};
use ny_core::dd_selfcheck::dd_selfcheck_ok;
use ny_core::f32_to_f64_exact;
use ny_core::Result;

use super::safe_mul_for_bounds_f64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutwardDirection {
    Lower,
    Upper,
}

/// Enclose `bias + Σ coefficient_i * value_i` in binary64.
///
/// Finite binary32 products are exact in binary64. A self-checked double-double
/// accumulation plus its certified gamma envelope keeps cancellation-heavy rows
/// tight. Non-finite inputs use the safe `0*±inf=0` convention and fall back to
/// directing every binary64 addition outward.
pub(crate) fn certified_affine_sum_f32<I>(bias: f32, terms: I, direction: OutwardDirection) -> f64
where
    I: IntoIterator<Item = (f32, f32)>,
{
    // #poll-amortise: `NoPoll` compiles to nothing, so the ubiquitous
    // no-deadline reduction keeps a hot loop with no counter, no branch and no
    // `Result` in it. Routing it through the counting adapter would tax every
    // caller of this wrapper. NOT every no-deadline concretization comes here:
    // `LinearBounds::concretize` reaches the POLLABLE form below through
    // `PatchesMaterializationDeadline::new(None)`, so it pays the per-term
    // counter and one flush -- two relaxed atomics on process-global counters --
    // per reduction, and those reductions also dominate the `poll_*` fields of
    // the `[poll-amortise]` observable.
    certified_affine_sum_f32_polled(bias, terms, direction, crate::amortise::NoPoll)
        .expect("the no-op certified-reduction poll cannot fail")
}

/// Pollable form of [`certified_affine_sum_f32`].
///
/// The poll is charged one unit per exact binary32 product folded into the
/// private reduction state, and DELIVERED once per
/// [`crate::amortise::POLL_CHUNK_UNITS`] units plus one final flush before
/// publication (#poll-amortise). A refusal still discards the whole private
/// state, so no partial endpoint can cross a proof boundary; the only change
/// is that expiry is noticed at most one stride -- 4096 MACs, microseconds --
/// later. This is the innermost loop of every CROWN concretization, so the
/// per-unit form put a foreign call and a `Result` branch in the hot kernel.
/// The no-op wrapper above keeps the ubiquitous no-deadline arithmetic and
/// iteration order byte-identical.
pub(crate) fn certified_affine_sum_f32_with_poll<I, P>(
    bias: f32,
    terms: I,
    direction: OutwardDirection,
    poll: P,
) -> Result<f64>
where
    I: IntoIterator<Item = (f32, f32)>,
    P: FnMut(usize) -> Result<()>,
{
    certified_affine_sum_f32_polled(
        bias,
        terms,
        direction,
        crate::amortise::ChunkedPoll::new(poll),
    )
}

/// The single certified-reduction kernel, generic over how it polls.
fn certified_affine_sum_f32_polled<I, R>(
    bias: f32,
    terms: I,
    direction: OutwardDirection,
    mut poll: R,
) -> Result<f64>
where
    I: IntoIterator<Item = (f32, f32)>,
    R: crate::amortise::ReductionPoll,
{
    let bias64 = f32_to_f64_exact(bias);
    let mut dd = Dd::from_f64(bias64);
    let mut abs_sum = bias64.abs();
    let mut directed = bias64;
    let mut count = 0usize;
    let mut dd_authorized = dd_selfcheck_ok() && bias.is_finite();
    let mut exact_zero = bias.to_bits() & 0x7fff_ffff == 0;

    for (coefficient, value) in terms {
        count = count.saturating_add(1);
        let coefficient64 = f32_to_f64_exact(coefficient);
        let value64 = f32_to_f64_exact(value);
        let product = safe_mul_for_bounds_f64(coefficient64, value64);
        let coefficient_is_zero = coefficient.to_bits() & 0x7fff_ffff == 0;
        let value_is_zero = value.to_bits() & 0x7fff_ffff == 0;
        exact_zero &= coefficient_is_zero || value_is_zero;
        directed = match direction {
            OutwardDirection::Lower => {
                let sum = directed + product;
                if sum.is_nan() {
                    f64::NEG_INFINITY
                } else {
                    next_down_f64(sum)
                }
            }
            OutwardDirection::Upper => {
                let sum = directed + product;
                if sum.is_nan() {
                    f64::INFINITY
                } else {
                    next_up_f64(sum)
                }
            }
        };

        if dd_authorized && coefficient.is_finite() && value.is_finite() {
            dd = dd_fma(dd, coefficient64, value64);
            let product_magnitude = product.abs();
            if product_magnitude != 0.0 {
                abs_sum = next_up_f64(abs_sum + product_magnitude);
            }
            dd_authorized = dd.is_finite() && abs_sum.is_finite();
        } else {
            dd_authorized = false;
        }
        poll.tick()?;
    }
    // Deliver the remainder BEFORE any publication path below, so a refusal
    // still lands ahead of the proof boundary.
    poll.flush()?;

    // A zero bias and all-zero products have exact real sum zero. Applying
    // the generic `next_*` publication below would invent a nonzero penalty
    // even though no arithmetic uncertainty exists. This matters when an
    // explicitly present but all-zero coefficient-error carrier is discharged:
    // it must be numerically identical to having no incoming error at all.
    // Inspect the source binary32 bits rather than a hardware conversion so a
    // DAZ host cannot misclassify a nonzero subnormal operand as zero.
    if exact_zero {
        return Ok(0.0);
    }

    if !dd_authorized {
        return Ok(directed);
    }

    let gamma = gamma_n_dd(count);
    let error = if gamma == 0.0 || abs_sum == 0.0 {
        0.0
    } else {
        next_up_f64(gamma * abs_sum)
    };
    Ok(match direction {
        OutwardDirection::Lower => {
            let represented = next_down_f64(dd.hi + dd.lo);
            next_down_f64(represented - error)
        }
        OutwardDirection::Upper => {
            let represented = next_up_f64(dd.hi + dd.lo);
            next_up_f64(represented + error)
        }
    })
}

#[cfg(test)]
mod tests {
    use ny_core::NyError;

    use super::*;

    #[test]
    fn cancellation_is_tight_and_outward() {
        let large = 2.0_f32.powi(30);
        let terms = [(large, large), (1.0, 1.0), (-large, large)];
        let lower = certified_affine_sum_f32(0.0, terms, OutwardDirection::Lower);
        let upper = certified_affine_sum_f32(0.0, terms, OutwardDirection::Upper);
        assert!(lower <= 1.0 && upper >= 1.0);
        assert!(lower > 0.99 && upper < 1.01, "[{lower:e}, {upper:e}]");
    }

    #[test]
    fn all_zero_reduction_is_exact_in_both_directions() {
        let terms = [
            (0.0, 3.0),
            (-0.0, -7.0),
            (5.0, 0.0),
            (0.0, f32::INFINITY),
            (f32::NEG_INFINITY, -0.0),
        ];
        assert_eq!(
            certified_affine_sum_f32(0.0, terms, OutwardDirection::Lower),
            0.0
        );
        assert_eq!(
            certified_affine_sum_f32(-0.0, terms, OutwardDirection::Upper),
            0.0
        );
    }

    #[test]
    fn nonzero_subnormal_operand_is_not_zero_fast_pathed() {
        let tiny = f32::from_bits(1);
        let exact = f32_to_f64_exact(tiny);
        let lower = certified_affine_sum_f32(0.0, [(tiny, 1.0)], OutwardDirection::Lower);
        let upper = certified_affine_sum_f32(0.0, [(tiny, 1.0)], OutwardDirection::Upper);
        assert!(lower > 0.0 && lower <= exact, "lower={lower:e}");
        assert!(upper >= exact, "upper={upper:e}");
    }

    #[test]
    fn pollable_reduction_matches_bits_and_refuses_before_publication() {
        let terms = [(3.0, -2.0), (-4.0, 0.5), (0.25, 8.0)];
        let expected = certified_affine_sum_f32(1.0, terms, OutwardDirection::Lower);
        let mut polls = 0usize;
        let actual =
            certified_affine_sum_f32_with_poll(1.0, terms, OutwardDirection::Lower, |units| {
                polls += units;
                Ok(())
            })
            .unwrap();
        assert_eq!(actual.to_bits(), expected.to_bits());
        assert_eq!(polls, terms.len());

        // #poll-amortise: three terms are far inside one stride, so they are
        // delivered in a single flush. The refusal must still land before the
        // reduction publishes anything.
        let mut injected_polls = 0usize;
        let error =
            certified_affine_sum_f32_with_poll(1.0, terms, OutwardDirection::Lower, |units| {
                injected_polls += units;
                if injected_polls >= 2 {
                    Err(NyError::DeadlineExceeded(
                        "injected certified reduction deadline".into(),
                    ))
                } else {
                    Ok(())
                }
            })
            .expect_err("injected poll must refuse the private reduction");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
        assert_eq!(injected_polls, terms.len());
    }

    #[test]
    fn chunked_reduction_poll_is_amortised_at_the_declared_stride() {
        let count = crate::amortise::POLL_CHUNK_UNITS * 2 + 7;
        let terms: Vec<(f32, f32)> = (0..count)
            .map(|index| match index % 3 {
                0 => (1.0_f32, 0.5_f32),
                1 => (-2.0_f32, 0.25_f32),
                _ => (0.125_f32, -4.0_f32),
            })
            .collect();

        let mut calls = 0usize;
        let mut units = 0usize;
        let amortised = certified_affine_sum_f32_with_poll(
            1.0,
            terms.iter().copied(),
            OutwardDirection::Lower,
            |charged| {
                calls += 1;
                units += charged;
                Ok(())
            },
        )
        .expect("no refusal");

        assert_eq!(units, count, "every MAC must still be charged to the poll");
        assert_eq!(calls, 3, "two full strides plus the remainder flush");

        let reference =
            certified_affine_sum_f32(1.0, terms.iter().copied(), OutwardDirection::Lower);
        assert_eq!(
            amortised.to_bits(),
            reference.to_bits(),
            "amortising the poll must not move a certified endpoint by one ulp"
        );
    }

    #[test]
    fn amortised_reduction_still_refuses_within_one_stride_of_expiry() {
        let count = crate::amortise::POLL_CHUNK_UNITS * 3;
        let terms: Vec<(f32, f32)> = (0..count).map(|_| (1.0_f32, 1.0_f32)).collect();
        let mut units = 0usize;
        let error = certified_affine_sum_f32_with_poll(
            0.0,
            terms.iter().copied(),
            OutwardDirection::Upper,
            |charged| {
                units += charged;
                Err(NyError::DeadlineExceeded(
                    "expired at the first stride".into(),
                ))
            },
        )
        .expect_err("the first stride boundary must refuse");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
        assert_eq!(
            units,
            crate::amortise::POLL_CHUNK_UNITS,
            "refusal must arrive at the first stride, not at the end of the reduction"
        );
    }
}
