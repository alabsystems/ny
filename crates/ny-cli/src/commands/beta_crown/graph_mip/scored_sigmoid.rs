// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proof-bearing scored-f32 Sigmoid bridge for the dormant dual-Sigmoid slice.
//!
//! This module deliberately does **not** claim that ONNX Runtime, CUDA, or any
//! other competition scoring backend implements the reference semantics below.
//! The only backend currently constructible here is named explicitly:
//! mathematical Sigmoid, rounded once to binary32 with IEEE round-to-nearest,
//! ties-to-even. A production caller must first establish that its scored
//! backend has exactly those semantics (or add a separately verified,
//! separately named evaluator). Backend proximity measurements are not proof.
//!
//! The reference evaluator contains no host `exp`, `log`, float-to-int
//! approximation, or empirical ULP allowance:
//!
//! * finite binary32 inputs are converted to exact [`BigRational`] values;
//! * the positive exponential is enclosed by a rational Taylor lower sum and
//!   a geometric upper remainder;
//! * both Sigmoid endpoints are rounded by exact comparisons against binary32
//!   midpoints;
//! * preimages are found by binary search over the total ordered binary32 bit
//!   domain, then widened to the adjacent input-rounding-cell midpoint.
//!
//! Saturated final brackets, non-finite values, an unclassified Taylor
//! enclosure, and any missing monotone bracket all decline. The returned types
//! have private fields and can only be minted by this verified evaluator.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{bail, Result};
use num_rational::BigRational;
use num_traits::{One, Zero};

const F32_SIGN: u32 = 0x8000_0000;
const F32_ONE_BITS: u32 = 0x3f80_0000;
const MIN_FINITE_KEY: u32 = 0x0080_0000; // ordered key of -f32::MAX
const MAX_FINITE_KEY: u32 = 0xff7f_ffff; // ordered key of +f32::MAX
const MAX_TAYLOR_TERMS: usize = 2048;

/// Exact scoring semantics to which an enclosure/preimage is qualified.
///
/// No ORT/CUDA variant exists yet: adding one requires a numerical contract or
/// a verified exact endpoint evaluator for that exact provider/version/kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScoredF32SigmoidBackend {
    /// `roundTiesToEven_f32(1 / (1 + exp(-x_f32)))`, with mathematical `exp`.
    CorrectlyRoundedRealV1,
}

/// Which monotone scored-output preimage was certified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScoredF32SigmoidPreimageSense {
    /// Every scored `sigmoid(x) >= threshold` input lies at or above `rhs`.
    AtLeast,
    /// Every scored `sigmoid(x) <= threshold` input lies at or below `rhs`.
    AtMost,
}

/// Certified enclosure of a scored binary32 Sigmoid output.
///
/// Endpoints are exact binary32 values widened losslessly into f64. Fields and
/// constructors are private: arbitrary numerical intervals cannot be promoted
/// to scored-backend provenance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PinnedSigmoidOutputEnclosure {
    backend: ScoredF32SigmoidBackend,
    lower: f64,
    upper: f64,
}

impl PinnedSigmoidOutputEnclosure {
    pub(super) fn backend(self) -> ScoredF32SigmoidBackend {
        self.backend
    }

    pub(super) fn lower(self) -> f64 {
        self.lower
    }

    pub(super) fn upper(self) -> f64 {
        self.upper
    }

    #[cfg(test)]
    pub(super) fn from_test_exact_endpoints(
        backend: ScoredF32SigmoidBackend,
        lower: f32,
        upper: f32,
    ) -> Result<Self> {
        if !lower.is_finite() || !upper.is_finite() || lower < 0.0 || upper > 1.0 || lower > upper {
            bail!(
                "test scored sigmoid enclosure must be finite and contained in [0, 1], got \
                 [{lower}, {upper}]"
            );
        }
        Ok(Self {
            backend,
            lower: f64::from(lower),
            upper: f64::from(upper),
        })
    }
}

/// A one-sided preimage minted by a backend-qualified exact evaluator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ScoredF32SigmoidPreimage {
    backend: ScoredF32SigmoidBackend,
    sense: ScoredF32SigmoidPreimageSense,
    threshold_bits: u64,
    rhs: f64,
    boundary_input: f32,
    outside_neighbor: f32,
}

impl ScoredF32SigmoidPreimage {
    pub(super) fn rhs_for(
        self,
        backend: ScoredF32SigmoidBackend,
        sense: ScoredF32SigmoidPreimageSense,
        threshold: f64,
    ) -> Result<f64> {
        if self.backend != backend
            || self.sense != sense
            || self.threshold_bits != threshold.to_bits()
            || !self.rhs.is_finite()
            || !self.boundary_input.is_finite()
            || !self.outside_neighbor.is_finite()
        {
            bail!("scored Sigmoid preimage provenance mismatch (fail closed)");
        }
        Ok(self.rhs)
    }
}

/// Exact evaluator and preimage factory for one explicitly named backend.
///
/// The cache contains only values recomputed by the rational evaluator; it is
/// an optimization and carries no authority of its own.
#[derive(Debug)]
pub(super) struct CertifiedScoredF32Sigmoid {
    backend: ScoredF32SigmoidBackend,
    cache: Mutex<HashMap<u32, f32>>,
}

impl CertifiedScoredF32Sigmoid {
    /// Construct the only currently verified semantics.
    ///
    /// This name is intentionally not `onnx_runtime` or `native`: callers may
    /// not silently treat the reference evaluator as evidence about another
    /// backend.
    pub(super) fn correctly_rounded_real_v1() -> Self {
        Self {
            backend: ScoredF32SigmoidBackend::CorrectlyRoundedRealV1,
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub(super) fn backend(&self) -> ScoredF32SigmoidBackend {
        self.backend
    }

    /// Evaluate a pinned binary32 logit and mint its exact scored enclosure.
    pub(super) fn pinned_output(&self, logit: f32) -> Result<PinnedSigmoidOutputEnclosure> {
        let output = self.evaluate(logit)?;
        Ok(PinnedSigmoidOutputEnclosure {
            backend: self.backend,
            lower: f64::from(output),
            upper: f64::from(output),
        })
    }

    /// Compute a certified monotone preimage by ordered-binary32 search.
    pub(super) fn preimage(
        &self,
        threshold: f64,
        sense: ScoredF32SigmoidPreimageSense,
    ) -> Result<ScoredF32SigmoidPreimage> {
        if !threshold.is_finite() || threshold <= 0.0 || threshold >= 1.0 {
            bail!(
                "scored Sigmoid preimage threshold must be finite and strictly inside (0, 1), \
                 got {threshold}"
            );
        }

        let mut low = MIN_FINITE_KEY;
        let mut high = MAX_FINITE_KEY;
        let low_output = f64::from(self.evaluate(float_from_ordered_key(low))?);
        let high_output = f64::from(self.evaluate(float_from_ordered_key(high))?);
        let (low_inside, high_inside) = match sense {
            ScoredF32SigmoidPreimageSense::AtLeast => {
                (low_output >= threshold, high_output >= threshold)
            }
            ScoredF32SigmoidPreimageSense::AtMost => {
                (low_output <= threshold, high_output <= threshold)
            }
        };
        let bracketed = match sense {
            ScoredF32SigmoidPreimageSense::AtLeast => !low_inside && high_inside,
            ScoredF32SigmoidPreimageSense::AtMost => low_inside && !high_inside,
        };
        if !bracketed {
            bail!(
                "scored Sigmoid preimage has no finite monotone bracket for threshold \
                 {threshold} (fail closed)"
            );
        }

        while high - low > 1 {
            let middle = low + (high - low) / 2;
            let output = f64::from(self.evaluate(float_from_ordered_key(middle))?);
            let inside = match sense {
                ScoredF32SigmoidPreimageSense::AtLeast => output >= threshold,
                ScoredF32SigmoidPreimageSense::AtMost => output <= threshold,
            };
            match sense {
                ScoredF32SigmoidPreimageSense::AtLeast if inside => high = middle,
                ScoredF32SigmoidPreimageSense::AtLeast => low = middle,
                ScoredF32SigmoidPreimageSense::AtMost if inside => low = middle,
                ScoredF32SigmoidPreimageSense::AtMost => high = middle,
            }
        }

        let (boundary_key, outside_key) = match sense {
            ScoredF32SigmoidPreimageSense::AtLeast => (high, low),
            ScoredF32SigmoidPreimageSense::AtMost => (low, high),
        };
        let boundary_input = float_from_ordered_key(boundary_key);
        let outside_neighbor = float_from_ordered_key(outside_key);
        let boundary_output = self.evaluate(boundary_input)?;
        let outside_output = self.evaluate(outside_neighbor)?;
        if boundary_key.abs_diff(outside_key) != 1
            || !boundary_input.is_finite()
            || !outside_neighbor.is_finite()
            || !boundary_output.is_finite()
            || !outside_output.is_finite()
            || boundary_output == 0.0
            || boundary_output == 1.0
            || outside_output == 0.0
            || outside_output == 1.0
        {
            bail!("scored Sigmoid preimage ended at a non-finite/saturated bracket (fail closed)");
        }
        let classification_is_exact = match sense {
            ScoredF32SigmoidPreimageSense::AtLeast => {
                f64::from(outside_output) < threshold
                    && f64::from(boundary_output) >= threshold
                    && outside_input_precedes(outside_neighbor, boundary_input)
            }
            ScoredF32SigmoidPreimageSense::AtMost => {
                f64::from(boundary_output) <= threshold
                    && f64::from(outside_output) > threshold
                    && outside_input_precedes(boundary_input, outside_neighbor)
            }
        };
        if !classification_is_exact {
            bail!("scored Sigmoid endpoint enclosure did not classify the adjacent bracket");
        }

        // Every real logit rounded to `boundary_input` lies within the cell
        // bounded by the exact midpoint of these adjacent f32 values. Widen
        // one f64 ULP in the never-stricter direction; no host transcendental
        // or platform float-to-int conversion is involved.
        let midpoint = (f64::from(boundary_input) + f64::from(outside_neighbor)) * 0.5;
        let rhs = match sense {
            ScoredF32SigmoidPreimageSense::AtLeast => midpoint.next_down(),
            ScoredF32SigmoidPreimageSense::AtMost => midpoint.next_up(),
        };
        if !midpoint.is_finite() || !rhs.is_finite() {
            bail!("scored Sigmoid input-cell midpoint is non-finite (fail closed)");
        }

        Ok(ScoredF32SigmoidPreimage {
            backend: self.backend,
            sense,
            threshold_bits: threshold.to_bits(),
            rhs,
            boundary_input,
            outside_neighbor,
        })
    }

    fn evaluate(&self, input: f32) -> Result<f32> {
        if !input.is_finite() {
            bail!("scored Sigmoid evaluator requires a finite f32 input");
        }
        let cached = {
            let cache = self
                .cache
                .lock()
                .map_err(|_| anyhow::anyhow!("scored Sigmoid cache lock poisoned"))?;
            cache.get(&input.to_bits()).copied()
        };
        if let Some(value) = cached {
            return Ok(value);
        }
        let value = correctly_rounded_real_sigmoid_f32(input)?;
        self.cache
            .lock()
            .map_err(|_| anyhow::anyhow!("scored Sigmoid cache lock poisoned"))?
            .insert(input.to_bits(), value);
        Ok(value)
    }
}

fn outside_input_precedes(lower: f32, upper: f32) -> bool {
    ordered_key(lower) < ordered_key(upper)
}

/// Monotone total-order key for finite IEEE binary32 values.
fn ordered_key(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & F32_SIGN == 0 {
        bits | F32_SIGN
    } else {
        !bits
    }
}

fn float_from_ordered_key(key: u32) -> f32 {
    let bits = if key & F32_SIGN == 0 {
        !key
    } else {
        key & !F32_SIGN
    };
    f32::from_bits(bits)
}

fn rational_from_usize(value: usize) -> Result<BigRational> {
    let value = i64::try_from(value)
        .map_err(|_| anyhow::anyhow!("rational Taylor index does not fit i64"))?;
    Ok(BigRational::from_integer(value.into()))
}

fn exact_f32_rational(value: f32) -> Result<BigRational> {
    BigRational::from_float(value)
        .ok_or_else(|| anyhow::anyhow!("cannot convert non-finite f32 {value} to exact rational"))
}

/// Round an exact rational in `[0,1]` to binary32, RN-ties-even, using only
/// integer/rational comparisons against exact adjacent-f32 midpoints.
fn round_unit_rational_to_f32(value: &BigRational) -> Result<f32> {
    if value < &BigRational::zero() || value > &BigRational::one() {
        bail!("unit rational round received value outside [0,1]");
    }
    let mut low = 0u32;
    let mut high = F32_ONE_BITS;
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        let middle_value = exact_f32_rational(f32::from_bits(middle))?;
        if middle_value <= *value {
            low = middle;
        } else {
            high = middle;
        }
    }
    let low_value = exact_f32_rational(f32::from_bits(low))?;
    if low_value == *value {
        return Ok(f32::from_bits(low));
    }
    let high_value = exact_f32_rational(f32::from_bits(high))?;
    let midpoint = (low_value + high_value) / rational_from_usize(2)?;
    Ok(if value < &midpoint {
        f32::from_bits(low)
    } else if value > &midpoint || low & 1 == 1 {
        f32::from_bits(high)
    } else {
        f32::from_bits(low)
    })
}

fn rational_pow(base: BigRational, exponent: usize) -> BigRational {
    let mut result = BigRational::one();
    for _ in 0..exponent {
        result *= &base;
    }
    result
}

/// Exact checks supporting the two saturation shortcuts.
///
/// `e > 27/10` follows from the positive Taylor terms through `1/4!`, hence
/// `e^18 > (27/10)^18 > 2^25-1`; this puts Sigmoid(18) above the binary32
/// midpoint immediately below 1. Likewise
/// `e > sum_{k=0}^8 1/k! = 109601/40320`, whose 104th power exceeds `2^150`;
/// therefore Sigmoid(-104) < 2^-150, the midpoint between 0 and the least
/// positive subnormal. All comparisons below are exact rationals.
fn saturation_cutoffs_proven() -> bool {
    static PROVEN: OnceLock<bool> = OnceLock::new();
    *PROVEN.get_or_init(|| {
        let two25_minus_one = BigRational::from_integer(((1i64 << 25) - 1).into());
        let positive_base = BigRational::new(27i64.into(), 10i64.into());
        let positive = rational_pow(positive_base, 18) > two25_minus_one;

        let negative_base = BigRational::new(109_601i64.into(), 40_320i64.into());
        let two150 = rational_pow(BigRational::from_integer(2i64.into()), 150);
        let negative = rational_pow(negative_base, 104) > two150;
        positive && negative
    })
}

/// Correctly rounded mathematical Sigmoid at one finite binary32 input.
fn correctly_rounded_real_sigmoid_f32(input: f32) -> Result<f32> {
    if !input.is_finite() {
        bail!("correctly-rounded Sigmoid requires finite input");
    }
    if input == 0.0 {
        return Ok(0.5);
    }
    if input >= 18.0 || input <= -104.0 {
        if !saturation_cutoffs_proven() {
            bail!("exact Sigmoid saturation cutoff proof failed");
        }
        return Ok(if input.is_sign_positive() { 1.0 } else { 0.0 });
    }

    let y = exact_f32_rational(input.abs())?;
    let one = BigRational::one();
    let mut term = one.clone();
    let mut sum = one.clone();
    for n in 1..=MAX_TAYLOR_TERMS {
        term *= &y;
        term /= rational_from_usize(n)?;
        sum += &term;

        // For S_n = sum_{k=0}^n y^k/k!, the positive remainder is at most
        // t_{n+1} / (1 - y/(n+2)) once y/(n+2) < 1.
        let mut next_term = term.clone();
        next_term *= &y;
        next_term /= rational_from_usize(n + 1)?;
        let ratio = &y / rational_from_usize(n + 2)?;
        if ratio >= one {
            continue;
        }
        let exp_lower = sum.clone();
        let exp_upper = &sum + next_term / (&one - ratio);
        let (sigmoid_lower, sigmoid_upper) = if input.is_sign_positive() {
            (
                &exp_lower / (&one + &exp_lower),
                &exp_upper / (&one + &exp_upper),
            )
        } else {
            (&one / (&one + &exp_upper), &one / (&one + &exp_lower))
        };
        let lower_rounded = round_unit_rational_to_f32(&sigmoid_lower)?;
        let upper_rounded = round_unit_rational_to_f32(&sigmoid_upper)?;
        if lower_rounded.to_bits() == upper_rounded.to_bits() {
            return Ok(lower_rounded);
        }
    }
    bail!(
        "exact rational Sigmoid enclosure did not classify f32 rounding after \
         {MAX_TAYLOR_TERMS} Taylor terms"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Static 250-decimal-digit references generated independently with
    /// Python Decimal's arbitrary-precision `exp`, then rounded by exact
    /// rational comparison to IEEE binary32 midpoints.
    #[test]
    fn exact_evaluator_matches_high_precision_boundary_references() {
        let references = [
            (0xc2d0_0000, 0x0000_0000), // -104 -> saturation to 0
            (0xc190_0000, 0x3282_d314), // -18
            (0xc180_0000, 0x33f1_aadc), // -16
            (0xbf80_0001, 0x3e89_b2b0), // next_down(-1)
            (0xbf80_0000, 0x3e89_b2b1), // -1
            (0xbf7f_ffff, 0x3e89_b2b1), // next_up(-1)
            (0x8000_0001, 0x3f00_0000), // next_down(-0)
            (0x8000_0000, 0x3f00_0000), // -0
            (0x0000_0000, 0x3f00_0000), // +0
            (0x0000_0001, 0x3f00_0000), // next_up(+0)
            (0x3f7f_ffff, 0x3f3b_26a7), // next_down(1)
            (0x3f80_0000, 0x3f3b_26a8), // 1
            (0x3f80_0001, 0x3f3b_26a8), // next_up(1)
            (0x4180_0000, 0x3f7f_fffe), // 16
            (0x4188_0000, 0x3f7f_ffff), // 17
            (0x418f_ffff, 0x3f80_0000), // next_down(18)
            (0x4190_0000, 0x3f80_0000), // 18 -> saturation to 1
        ];
        for (input_bits, expected_bits) in references {
            let actual = correctly_rounded_real_sigmoid_f32(f32::from_bits(input_bits))
                .unwrap_or_else(|e| panic!("input 0x{input_bits:08x}: {e:#}"));
            assert_eq!(actual.to_bits(), expected_bits, "input 0x{input_bits:08x}");
        }
    }

    #[test]
    fn ordered_f32_key_roundtrips_and_orders_signed_zero() {
        let values = [
            -f32::MAX,
            -1.0,
            f32::from_bits(0x8000_0001),
            -0.0,
            0.0,
            f32::from_bits(1),
            1.0,
            f32::MAX,
        ];
        for &value in &values {
            assert_eq!(
                float_from_ordered_key(ordered_key(value)).to_bits(),
                value.to_bits()
            );
        }
        for pair in values.windows(2) {
            assert!(ordered_key(pair[0]) < ordered_key(pair[1]));
        }
    }

    #[test]
    fn preimage_is_adjacent_outward_and_backend_bound() {
        let evaluator = CertifiedScoredF32Sigmoid::correctly_rounded_real_v1();
        for (sense, threshold) in [
            (ScoredF32SigmoidPreimageSense::AtLeast, 0.75),
            (ScoredF32SigmoidPreimageSense::AtMost, 0.25),
            (
                ScoredF32SigmoidPreimageSense::AtLeast,
                f64::from(f32::from_bits(0x3f00_0001)),
            ),
            (
                ScoredF32SigmoidPreimageSense::AtMost,
                f64::from(f32::from_bits(0x3eff_ffff)),
            ),
        ] {
            let proof = evaluator.preimage(threshold, sense).unwrap();
            let rhs = proof
                .rhs_for(evaluator.backend(), sense, threshold)
                .unwrap();
            assert!(rhs.is_finite());
            assert_eq!(
                ordered_key(proof.boundary_input).abs_diff(ordered_key(proof.outside_neighbor)),
                1
            );
            let boundary = evaluator.evaluate(proof.boundary_input).unwrap();
            let outside = evaluator.evaluate(proof.outside_neighbor).unwrap();
            let input_midpoint =
                (f64::from(proof.boundary_input) + f64::from(proof.outside_neighbor)) * 0.5;
            match sense {
                ScoredF32SigmoidPreimageSense::AtLeast => {
                    assert!(f64::from(outside) < threshold);
                    assert!(f64::from(boundary) >= threshold);
                    assert!(rhs < input_midpoint, "lower rhs must widen downward");
                }
                ScoredF32SigmoidPreimageSense::AtMost => {
                    assert!(f64::from(boundary) <= threshold);
                    assert!(f64::from(outside) > threshold);
                    assert!(rhs > input_midpoint, "upper rhs must widen upward");
                }
            }
            assert!(proof
                .rhs_for(
                    evaluator.backend(),
                    sense,
                    f64::from_bits(threshold.to_bits() ^ 1)
                )
                .is_err());
        }
    }

    #[test]
    fn nonfinite_boundary_and_saturation_requests_decline() {
        let evaluator = CertifiedScoredF32Sigmoid::correctly_rounded_real_v1();
        for threshold in [
            f64::NAN,
            f64::NEG_INFINITY,
            f64::INFINITY,
            -f64::EPSILON,
            0.0,
            1.0,
            1.0 + f64::EPSILON,
        ] {
            assert!(evaluator
                .preimage(threshold, ScoredF32SigmoidPreimageSense::AtLeast)
                .is_err());
            assert!(evaluator
                .preimage(threshold, ScoredF32SigmoidPreimageSense::AtMost)
                .is_err());
        }
        assert!(correctly_rounded_real_sigmoid_f32(f32::NAN).is_err());
        assert!(correctly_rounded_real_sigmoid_f32(f32::INFINITY).is_err());
        assert!(correctly_rounded_real_sigmoid_f32(f32::NEG_INFINITY).is_err());

        // These thresholds place the adjacent crossing at a scored saturation
        // plateau (0 or 1), which the preimage API intentionally refuses.
        assert!(evaluator
            .preimage(
                f64::from(f32::from_bits(1)),
                ScoredF32SigmoidPreimageSense::AtLeast
            )
            .is_err());
        assert!(evaluator
            .preimage(
                f64::from(f32::from_bits(0x3f7f_ffff)),
                ScoredF32SigmoidPreimageSense::AtMost
            )
            .is_err());
    }

    #[test]
    fn exact_rational_saturation_inequalities_hold() {
        assert!(saturation_cutoffs_proven());
    }
}
