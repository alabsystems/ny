// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Convex mixtures of retained affine lower certificates.
//!
//! A [`FacetBank`] belongs to one scalar specification row.  If every retained
//! affine form `l_t(x)` is a lower bound on that same scalar objective over the
//! same domain, then every simplex mixture is also a lower bound:
//!
//! `sum_t lambda_t l_t(x) <= f(x)` for `lambda_t >= 0` and `sum_t lambda_t = 1`.
//!
//! Concretizing a mixture can be strictly tighter than taking the best
//! concretized plane separately. For example, the valid lower planes `x` and
//! `-x` for the objective `|x| = ReLU(x) + ReLU(-x)` on `[-1, 1]` each
//! concretize to `-1`, while their half/half mixture is zero.
//!
//! This module is deliberately independent of alpha/beta optimization.  The
//! optimizer may propose planes and mixture weights, but verdict-grade bounds
//! are produced only by the fail-closed interval arithmetic below.

use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, BoundedTensor};

/// Hard memory/latency cap for one scalar row's retained affine facets.
pub const FACET_BANK_MAX_PLANES: usize = 8;

/// Default denominator is `2^12`, giving fine weights without a large search.
pub const FACET_BANK_DEFAULT_DYADIC_BITS: u8 = 12;

const FACET_BANK_MAX_DYADIC_BITS: u8 = 20;
const FACET_BANK_MAX_REFINEMENT_ROUNDS: usize = 64;

/// One sound lower-affine certificate for one scalar specification row.
///
/// The represented exact affine form has coefficient vector `a_star` and bias
/// `b_star` satisfying
///
/// `|a_star[j] - coefficients[j]| <= coefficient_errors[j]` and
/// `|b_star - bias| <= bias_error`.
///
/// That exact affine form must already be known to lower-bound the objective.
/// FacetBank discharges the stored numerical errors outward over the supplied
/// input box; it does not establish the semantic validity of a newly invented
/// plane.
#[derive(Clone, Debug, PartialEq)]
pub struct LowerAffineCertificate {
    coefficients: Vec<f32>,
    coefficient_errors: Option<Vec<f32>>,
    bias: f32,
    bias_error: f32,
}

impl LowerAffineCertificate {
    /// Construct a certificate whose stored coefficients and bias are exact.
    pub fn new(coefficients: Vec<f32>, bias: f32) -> Result<Self> {
        Self::from_parts(coefficients, bias, None, 0.0)
    }

    /// Construct a certificate with symmetric coefficient and bias errors.
    pub fn with_errors(
        coefficients: Vec<f32>,
        bias: f32,
        coefficient_errors: Vec<f32>,
        bias_error: f32,
    ) -> Result<Self> {
        Self::from_parts(coefficients, bias, Some(coefficient_errors), bias_error)
    }

    /// Construct a certificate from optional per-coefficient error storage.
    ///
    /// This matches GPU resident coefficient carriers, where the error vector
    /// may be absent (meaning exact/zero error) or present for every input
    /// coefficient.
    pub fn from_parts(
        coefficients: Vec<f32>,
        bias: f32,
        coefficient_errors: Option<Vec<f32>>,
        bias_error: f32,
    ) -> Result<Self> {
        if coefficients.is_empty() {
            return Err(NyError::InvalidSpec(
                "FacetBank certificate must have at least one coefficient".into(),
            ));
        }
        if coefficients.iter().any(|v| !v.is_finite()) || !bias.is_finite() {
            return Err(NyError::NumericalInstability(
                "FacetBank certificate contains a non-finite coefficient or bias".into(),
            ));
        }
        if !bias_error.is_finite() || bias_error < 0.0 {
            return Err(NyError::NumericalInstability(
                "FacetBank bias error must be finite and non-negative".into(),
            ));
        }
        if let Some(errors) = coefficient_errors.as_ref() {
            if errors.len() != coefficients.len() {
                return Err(NyError::shape_mismatch(
                    vec![coefficients.len()],
                    vec![errors.len()],
                ));
            }
            if errors.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(NyError::NumericalInstability(
                    "FacetBank coefficient errors must be finite and non-negative".into(),
                ));
            }
        }
        Ok(Self {
            coefficients,
            coefficient_errors,
            bias,
            bias_error,
        })
    }

    /// Stored center coefficients.
    #[inline]
    pub fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }

    /// Optional symmetric errors on the stored coefficients.
    #[inline]
    pub fn coefficient_errors(&self) -> Option<&[f32]> {
        self.coefficient_errors.as_deref()
    }

    /// Stored center bias.
    #[inline]
    pub fn bias(&self) -> f32 {
        self.bias
    }

    /// Symmetric error on the stored bias.
    #[inline]
    pub fn bias_error(&self) -> f32 {
        self.bias_error
    }

    #[inline]
    fn coefficient_error(&self, index: usize) -> f32 {
        self.coefficient_errors
            .as_ref()
            .map_or(0.0, |errors| errors[index])
    }
}

/// Deterministic dyadic-mixture search settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FacetBankSearchConfig {
    /// Mixture weights have the exact denominator `2^dyadic_bits`.
    pub dyadic_bits: u8,
    /// Maximum deterministic supergradient refinement rounds.
    pub refinement_rounds: usize,
}

impl Default for FacetBankSearchConfig {
    fn default() -> Self {
        Self {
            dyadic_bits: FACET_BANK_DEFAULT_DYADIC_BITS,
            refinement_rounds: 8,
        }
    }
}

impl FacetBankSearchConfig {
    fn validate(self) -> Result<Self> {
        if self.dyadic_bits == 0 || self.dyadic_bits > FACET_BANK_MAX_DYADIC_BITS {
            return Err(NyError::InvalidSpec(format!(
                "FacetBank dyadic_bits must be in 1..={FACET_BANK_MAX_DYADIC_BITS}, got {}",
                self.dyadic_bits
            )));
        }
        if self.refinement_rounds > FACET_BANK_MAX_REFINEMENT_ROUNDS {
            return Err(NyError::InvalidSpec(format!(
                "FacetBank refinement_rounds must be <= {FACET_BANK_MAX_REFINEMENT_ROUNDS}, got {}",
                self.refinement_rounds
            )));
        }
        Ok(self)
    }
}

/// Result of certifying one scalar row's retained facets.
#[derive(Clone, Debug, PartialEq)]
pub struct FacetBankBound {
    /// Best certified lower bound, rounded toward negative infinity to `f32`.
    pub lower_bound: f32,
    /// Best independently concretized retained plane, in the same rounding mode.
    pub best_one_hot: f32,
    /// Numerators of the selected exact dyadic simplex weights.
    pub weight_numerators: Vec<u32>,
    /// Common exact denominator of `weight_numerators`.
    pub weight_denominator: u32,
    /// Index of the strongest independently concretized retained plane.
    pub best_one_hot_index: usize,
}

/// Retained lower-affine facets for one scalar objective/specification row.
#[derive(Clone, Debug)]
pub struct FacetBank {
    input_dim: usize,
    planes: Vec<LowerAffineCertificate>,
    config: FacetBankSearchConfig,
}

impl FacetBank {
    /// Create an empty bank for a scalar row over `input_dim` input coordinates.
    pub fn new(input_dim: usize) -> Result<Self> {
        Self::with_config(input_dim, FacetBankSearchConfig::default())
    }

    /// Create an empty bank with explicit deterministic search settings.
    pub fn with_config(input_dim: usize, config: FacetBankSearchConfig) -> Result<Self> {
        if input_dim == 0 {
            return Err(NyError::InvalidSpec(
                "FacetBank input dimension must be non-zero".into(),
            ));
        }
        Ok(Self {
            input_dim,
            planes: Vec::new(),
            config: config.validate()?,
        })
    }

    /// Build a bank from already-validated row certificates.
    pub fn from_certificates(
        certificates: Vec<LowerAffineCertificate>,
        config: FacetBankSearchConfig,
    ) -> Result<Self> {
        let input_dim = certificates.first().map_or(0, |p| p.coefficients.len());
        let mut bank = Self::with_config(input_dim, config)?;
        for certificate in certificates {
            bank.push(certificate)?;
        }
        Ok(bank)
    }

    /// Retain another certificate, refusing dimension mismatches or a ninth plane.
    pub fn push(&mut self, certificate: LowerAffineCertificate) -> Result<()> {
        if certificate.coefficients.len() != self.input_dim {
            return Err(NyError::shape_mismatch(
                vec![self.input_dim],
                vec![certificate.coefficients.len()],
            ));
        }
        if self.planes.len() >= FACET_BANK_MAX_PLANES {
            return Err(NyError::InvalidSpec(format!(
                "FacetBank holds at most {FACET_BANK_MAX_PLANES} planes per scalar row"
            )));
        }
        self.planes.push(certificate);
        Ok(())
    }

    /// Number of retained planes.
    #[inline]
    pub fn len(&self) -> usize {
        self.planes.len()
    }

    /// Whether the bank contains no retained plane.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.planes.is_empty()
    }

    /// Input dimension shared by all retained planes.
    #[inline]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Read-only retained certificates.
    #[inline]
    pub fn certificates(&self) -> &[LowerAffineCertificate] {
        &self.planes
    }

    /// Search deterministic dyadic mixtures and return a certified lower bound.
    ///
    /// Every one-hot weight vector is evaluated first.  The final result is an
    /// explicit maximum with that baseline, so mixture-search quality can never
    /// make the returned bound weaker than the best retained plane under this
    /// module's verdict-grade concretization.
    pub fn certify(&self, input: &BoundedTensor) -> Result<FacetBankBound> {
        if self.planes.is_empty() {
            return Err(NyError::InvalidSpec(
                "cannot certify an empty FacetBank".into(),
            ));
        }
        let domain = ValidatedBox::new(input, self.input_dim)?;
        let denominator = 1_u32 << self.config.dyadic_bits;

        let mut best_one_hot_value = f64::NEG_INFINITY;
        let mut best_one_hot_index = 0_usize;
        for index in 0..self.planes.len() {
            let value = certify_single_plane(&self.planes[index], &domain)?;
            if value > best_one_hot_value {
                best_one_hot_value = value;
                best_one_hot_index = index;
            }
        }

        let mut best_value = best_one_hot_value;
        let mut best_weights = one_hot_weights(self.planes.len(), best_one_hot_index, denominator);

        // The balanced all-plane point catches genuinely multi-facet optima cheaply.
        let uniform = balanced_uniform_weights(self.planes.len(), denominator);
        consider_candidate(
            &self.planes,
            &domain,
            denominator,
            uniform,
            &mut best_value,
            &mut best_weights,
        )?;

        // Pair midpoints are especially valuable and lock in the canonical
        // x/-x strict-gain case without relying on a gradient convention at zero.
        let half = denominator / 2;
        for left in 0..self.planes.len() {
            for right in (left + 1)..self.planes.len() {
                let mut weights = vec![0_u32; self.planes.len()];
                weights[left] = half;
                weights[right] = denominator - half;
                consider_candidate(
                    &self.planes,
                    &domain,
                    denominator,
                    weights,
                    &mut best_value,
                    &mut best_weights,
                )?;
            }
        }

        // Deterministic Frank-Wolfe-like refinement.  It only proposes exact
        // dyadic candidates; each proposal is independently certified below.
        for _ in 0..self.config.refinement_rounds {
            let targets =
                top_supergradient_targets(&self.planes, &domain, &best_weights, denominator);
            let mut round_value = best_value;
            let mut round_weights = best_weights.clone();
            let mut steps = vec![
                denominator / 2,
                denominator / 4,
                denominator / 8,
                denominator / 16,
                1,
            ];
            steps.retain(|step| *step > 0);
            steps.sort_unstable();
            steps.dedup();
            for target in targets {
                for &step in &steps {
                    let candidate =
                        dyadic_step_toward_vertex(&best_weights, target, step, denominator);
                    consider_candidate(
                        &self.planes,
                        &domain,
                        denominator,
                        candidate,
                        &mut round_value,
                        &mut round_weights,
                    )?;
                }
            }
            if round_value > best_value {
                best_value = round_value;
                best_weights = round_weights;
            } else {
                break;
            }
        }

        let best_one_hot = lower_f64_to_f32(best_one_hot_value)?;
        let searched = lower_f64_to_f32(best_value)?;
        // Defense in depth: monotonic conversion should already preserve this,
        // but make the no-regression contract explicit at the API boundary.
        let lower_bound = searched.max(best_one_hot);
        if lower_bound == best_one_hot && searched < best_one_hot {
            best_weights = one_hot_weights(self.planes.len(), best_one_hot_index, denominator);
        }

        Ok(FacetBankBound {
            lower_bound,
            best_one_hot,
            weight_numerators: best_weights,
            weight_denominator: denominator,
            best_one_hot_index,
        })
    }

    /// Certify caller-supplied exact dyadic simplex weights without searching.
    ///
    /// This is useful for replaying a persisted FacetBank certificate.  Negative
    /// weights are unrepresentable, and the integer numerators must sum exactly
    /// to `2^dyadic_bits`.
    pub fn certify_dyadic_mixture(
        &self,
        input: &BoundedTensor,
        weight_numerators: &[u32],
        dyadic_bits: u8,
    ) -> Result<f32> {
        if self.planes.is_empty() {
            return Err(NyError::InvalidSpec(
                "cannot certify an empty FacetBank".into(),
            ));
        }
        if dyadic_bits == 0 || dyadic_bits > FACET_BANK_MAX_DYADIC_BITS {
            return Err(NyError::InvalidSpec(format!(
                "FacetBank dyadic_bits must be in 1..={FACET_BANK_MAX_DYADIC_BITS}, got {dyadic_bits}"
            )));
        }
        let domain = ValidatedBox::new(input, self.input_dim)?;
        let denominator = 1_u32 << dyadic_bits;
        let value = certify_weights_f64(&self.planes, &domain, weight_numerators, denominator)?;
        lower_f64_to_f32(value)
    }
}

#[derive(Clone, Debug)]
struct ValidatedBox {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl ValidatedBox {
    fn new(input: &BoundedTensor, expected_dim: usize) -> Result<Self> {
        if input.len() != expected_dim {
            return Err(NyError::shape_mismatch(
                vec![expected_dim],
                vec![input.len()],
            ));
        }
        let mut lower = Vec::with_capacity(expected_dim);
        let mut upper = Vec::with_capacity(expected_dim);
        for (&lo, &hi) in input.lower().iter().zip(input.upper().iter()) {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(NyError::NumericalInstability(
                    "FacetBank input box contains a non-finite endpoint".into(),
                ));
            }
            if lo > hi {
                return Err(NyError::InvalidSpec(
                    "FacetBank input box contains an inverted interval".into(),
                ));
            }
            lower.push(f64::from(lo));
            upper.push(f64::from(hi));
        }
        Ok(Self { lower, upper })
    }
}

fn one_hot_weights(count: usize, index: usize, denominator: u32) -> Vec<u32> {
    let mut weights = vec![0_u32; count];
    weights[index] = denominator;
    weights
}

fn balanced_uniform_weights(count: usize, denominator: u32) -> Vec<u32> {
    let count_u32 = u32::try_from(count).expect("FacetBank plane cap fits u32");
    let base = denominator / count_u32;
    let remainder = denominator % count_u32;
    (0..count)
        .map(|index| base + u32::from((index as u32) < remainder))
        .collect()
}

fn consider_candidate(
    planes: &[LowerAffineCertificate],
    domain: &ValidatedBox,
    denominator: u32,
    candidate: Vec<u32>,
    best_value: &mut f64,
    best_weights: &mut Vec<u32>,
) -> Result<()> {
    if candidate == *best_weights {
        return Ok(());
    }
    let value = certify_weights_f64(planes, domain, &candidate, denominator)?;
    if value > *best_value {
        *best_value = value;
        *best_weights = candidate;
    }
    Ok(())
}

fn validate_weights(weights: &[u32], plane_count: usize, denominator: u32) -> Result<()> {
    if weights.len() != plane_count {
        return Err(NyError::shape_mismatch(
            vec![plane_count],
            vec![weights.len()],
        ));
    }
    let sum: u64 = weights.iter().map(|&weight| u64::from(weight)).sum();
    if sum != u64::from(denominator) {
        return Err(NyError::InvalidSpec(format!(
            "FacetBank dyadic weight numerators sum to {sum}, expected {denominator}"
        )));
    }
    Ok(())
}

fn certify_weights_f64(
    planes: &[LowerAffineCertificate],
    domain: &ValidatedBox,
    weights: &[u32],
    denominator: u32,
) -> Result<f64> {
    validate_weights(weights, planes.len(), denominator)?;

    // Preserve the direct one-plane certificate exactly (up to its necessary
    // interval operations), rather than introducing needless lambda arithmetic.
    if let Some(index) = weights.iter().position(|&weight| weight == denominator) {
        if weights
            .iter()
            .enumerate()
            .all(|(other, &weight)| other == index || weight == 0)
        {
            return certify_single_plane(&planes[index], domain);
        }
    }

    let mut bias_interval = IntervalAccumulator::default();
    for (plane, &numerator) in planes.iter().zip(weights) {
        if numerator == 0 {
            continue;
        }
        let bias = center_error_interval(f64::from(plane.bias), f64::from(plane.bias_error))?;
        bias_interval.add_scaled(bias, numerator, denominator)?;
    }
    let (bias_lower, _) = bias_interval.finish()?;
    let mut total = bias_lower;

    for index in 0..domain.lower.len() {
        let mut coefficient_interval = IntervalAccumulator::default();
        for (plane, &numerator) in planes.iter().zip(weights) {
            if numerator == 0 {
                continue;
            }
            let coefficient = center_error_interval(
                f64::from(plane.coefficients[index]),
                f64::from(plane.coefficient_error(index)),
            )?;
            coefficient_interval.add_scaled(coefficient, numerator, denominator)?;
        }
        let coefficient = coefficient_interval.finish()?;
        let contribution =
            interval_product_lower(coefficient, (domain.lower[index], domain.upper[index]))?;
        total = add_down(total, contribution)?;
    }
    if total.is_nan() {
        return Err(NyError::NumericalInstability(
            "NaN while certifying FacetBank mixture".into(),
        ));
    }
    Ok(total)
}

fn certify_single_plane(plane: &LowerAffineCertificate, domain: &ValidatedBox) -> Result<f64> {
    let (mut total, _) = center_error_interval(f64::from(plane.bias), f64::from(plane.bias_error))?;
    for index in 0..domain.lower.len() {
        let coefficient = center_error_interval(
            f64::from(plane.coefficients[index]),
            f64::from(plane.coefficient_error(index)),
        )?;
        let contribution =
            interval_product_lower(coefficient, (domain.lower[index], domain.upper[index]))?;
        total = add_down(total, contribution)?;
    }
    if total.is_nan() {
        return Err(NyError::NumericalInstability(
            "NaN while certifying FacetBank plane".into(),
        ));
    }
    Ok(total)
}

#[derive(Default)]
struct IntervalAccumulator {
    lower: f64,
    upper: f64,
    seen: bool,
}

impl IntervalAccumulator {
    fn add_scaled(&mut self, interval: (f64, f64), numerator: u32, denominator: u32) -> Result<()> {
        debug_assert!(numerator > 0);
        let scaled = if numerator == denominator {
            interval
        } else {
            let weight = f64::from(numerator) / f64::from(denominator);
            (mul_down(weight, interval.0)?, mul_up(weight, interval.1)?)
        };
        if self.seen {
            self.lower = add_down(self.lower, scaled.0)?;
            self.upper = add_up(self.upper, scaled.1)?;
        } else {
            self.lower = scaled.0;
            self.upper = scaled.1;
            self.seen = true;
        }
        Ok(())
    }

    fn finish(self) -> Result<(f64, f64)> {
        if !self.seen || self.lower.is_nan() || self.upper.is_nan() || self.lower > self.upper {
            return Err(NyError::NumericalInstability(
                "invalid FacetBank interval accumulator".into(),
            ));
        }
        Ok((self.lower, self.upper))
    }
}

fn center_error_interval(center: f64, error: f64) -> Result<(f64, f64)> {
    debug_assert!(center.is_finite());
    debug_assert!(error.is_finite() && error >= 0.0);
    if error == 0.0 {
        return Ok((center, center));
    }
    let lower = sub_down(center, error)?;
    let upper = add_up(center, error)?;
    if lower > upper {
        return Err(NyError::NumericalInstability(
            "inverted FacetBank center/error interval".into(),
        ));
    }
    Ok((lower, upper))
}

fn interval_product_lower(left: (f64, f64), right: (f64, f64)) -> Result<f64> {
    let products = [
        mul_down(left.0, right.0)?,
        mul_down(left.0, right.1)?,
        mul_down(left.1, right.0)?,
        mul_down(left.1, right.1)?,
    ];
    products
        .into_iter()
        .reduce(f64::min)
        .ok_or_else(|| NyError::NumericalInstability("empty FacetBank product".into()))
}

fn top_supergradient_targets(
    planes: &[LowerAffineCertificate],
    domain: &ValidatedBox,
    weights: &[u32],
    denominator: u32,
) -> Vec<usize> {
    let inverse_denominator = 1.0 / f64::from(denominator);
    let max_abs: Vec<f64> = domain
        .lower
        .iter()
        .zip(&domain.upper)
        .map(|(&lo, &hi)| lo.abs().max(hi.abs()))
        .collect();
    let mut mixed_coefficients = vec![0.0_f64; domain.lower.len()];
    for (plane, &numerator) in planes.iter().zip(weights) {
        let weight = f64::from(numerator) * inverse_denominator;
        for (mixed, &coefficient) in mixed_coefficients.iter_mut().zip(&plane.coefficients) {
            *mixed += weight * f64::from(coefficient);
        }
    }
    let corner: Vec<f64> = mixed_coefficients
        .iter()
        .enumerate()
        .map(|(index, &coefficient)| {
            if coefficient >= 0.0 {
                domain.lower[index]
            } else {
                domain.upper[index]
            }
        })
        .collect();

    let mut scores: Vec<(usize, f64)> = planes
        .iter()
        .enumerate()
        .map(|(plane_index, plane)| {
            // Fold the symmetric stored errors over this box for the search
            // oracle only.  Rounding here cannot affect soundness because the
            // selected integer weights are re-certified independently.
            let coefficient_penalty: f64 = (0..domain.lower.len())
                .map(|index| f64::from(plane.coefficient_error(index)) * max_abs[index])
                .sum();
            let mut score =
                f64::from(plane.bias) - f64::from(plane.bias_error) - coefficient_penalty;
            for (&coefficient, &x) in plane.coefficients.iter().zip(&corner) {
                score += f64::from(coefficient) * x;
            }
            (plane_index, score)
        })
        .collect();
    scores.sort_by(|(left_index, left), (right_index, right)| {
        right
            .total_cmp(left)
            .then_with(|| left_index.cmp(right_index))
    });
    scores
        .into_iter()
        .take(planes.len().min(2))
        .map(|(index, _)| index)
        .collect()
}

fn dyadic_step_toward_vertex(
    weights: &[u32],
    target: usize,
    step: u32,
    denominator: u32,
) -> Vec<u32> {
    debug_assert!(target < weights.len());
    debug_assert!(step <= denominator);
    let keep = denominator - step;
    let denominator_u64 = u64::from(denominator);
    let mut result = vec![0_u32; weights.len()];
    let mut remainders = Vec::with_capacity(weights.len());
    let mut assigned = 0_u32;
    for (index, &weight) in weights.iter().enumerate() {
        let numerator = u64::from(weight) * u64::from(keep)
            + if index == target {
                u64::from(step) * denominator_u64
            } else {
                0
            };
        let quotient = u32::try_from(numerator / denominator_u64)
            .expect("dyadic quotient is bounded by denominator");
        let remainder = numerator % denominator_u64;
        result[index] = quotient;
        assigned += quotient;
        remainders.push((index, remainder));
    }
    let missing = denominator - assigned;
    remainders.sort_by(|(left_index, left), (right_index, right)| {
        right.cmp(left).then_with(|| left_index.cmp(right_index))
    });
    for &(index, _) in remainders.iter().take(missing as usize) {
        result[index] += 1;
    }
    debug_assert_eq!(result.iter().copied().sum::<u32>(), denominator);
    result
}

#[inline]
fn next_up_f64(value: f64) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude >= f64::INFINITY.to_bits() {
        return value;
    }
    if magnitude == 0 {
        return f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

#[inline]
fn next_down_f64(value: f64) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude >= f64::INFINITY.to_bits() {
        return value;
    }
    if magnitude == 0 {
        return -f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

fn add_down(left: f64, right: f64) -> Result<f64> {
    let value = left + right;
    if value.is_nan() {
        Err(NyError::NumericalInstability(
            "NaN in FacetBank downward addition".into(),
        ))
    } else {
        Ok(next_down_f64(value))
    }
}

fn add_up(left: f64, right: f64) -> Result<f64> {
    let value = left + right;
    if value.is_nan() {
        Err(NyError::NumericalInstability(
            "NaN in FacetBank upward addition".into(),
        ))
    } else {
        Ok(next_up_f64(value))
    }
}

fn sub_down(left: f64, right: f64) -> Result<f64> {
    let value = left - right;
    if value.is_nan() {
        Err(NyError::NumericalInstability(
            "NaN in FacetBank downward subtraction".into(),
        ))
    } else {
        Ok(next_down_f64(value))
    }
}

fn mul_down(left: f64, right: f64) -> Result<f64> {
    let value = left * right;
    if value.is_nan() {
        Err(NyError::NumericalInstability(
            "NaN in FacetBank downward multiplication".into(),
        ))
    } else {
        Ok(next_down_f64(value))
    }
}

fn mul_up(left: f64, right: f64) -> Result<f64> {
    let value = left * right;
    if value.is_nan() {
        Err(NyError::NumericalInstability(
            "NaN in FacetBank upward multiplication".into(),
        ))
    } else {
        Ok(next_up_f64(value))
    }
}

fn lower_f64_to_f32(value: f64) -> Result<f32> {
    if value.is_nan() {
        return Err(NyError::NumericalInstability(
            "NaN FacetBank lower bound".into(),
        ));
    }
    if value == f64::NEG_INFINITY || value < -f64::from(f32::MAX) {
        return Ok(f32::NEG_INFINITY);
    }
    if value == f64::INFINITY || value > f64::from(f32::MAX) {
        return Ok(f32::MAX);
    }
    let nearest = value as f32;
    if f64::from(nearest) > value {
        Ok(next_down_f32(nearest))
    } else {
        Ok(nearest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use num_bigint::BigInt;
    use num_rational::BigRational;

    fn box_1d(lower: f32, upper: f32) -> BoundedTensor {
        BoundedTensor::new(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn()).unwrap()
    }

    fn rational_f32(value: f32) -> BigRational {
        BigRational::from_float(value).expect("finite f32")
    }

    fn rational_weight(numerator: u32, denominator: u32) -> BigRational {
        BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
    }

    fn exact_interval_mixture_min(
        planes: &[LowerAffineCertificate],
        domain: &ValidatedBox,
        weights: &[u32],
        denominator: u32,
    ) -> BigRational {
        let mut total = BigRational::from_integer(BigInt::from(0));
        for (plane, &numerator) in planes.iter().zip(weights) {
            if numerator == 0 {
                continue;
            }
            let weight = rational_weight(numerator, denominator);
            total += weight * (rational_f32(plane.bias) - rational_f32(plane.bias_error));
        }
        for index in 0..domain.lower.len() {
            let mut coefficient_lower = BigRational::from_integer(BigInt::from(0));
            let mut coefficient_upper = BigRational::from_integer(BigInt::from(0));
            for (plane, &numerator) in planes.iter().zip(weights) {
                if numerator == 0 {
                    continue;
                }
                let weight = rational_weight(numerator, denominator);
                let center = rational_f32(plane.coefficients[index]);
                let error = rational_f32(plane.coefficient_error(index));
                coefficient_lower += weight.clone() * (center.clone() - error.clone());
                coefficient_upper += weight * (center + error);
            }
            let lo = BigRational::from_float(domain.lower[index]).unwrap();
            let hi = BigRational::from_float(domain.upper[index]).unwrap();
            let products = [
                coefficient_lower.clone() * lo.clone(),
                coefficient_lower * hi.clone(),
                coefficient_upper.clone() * lo,
                coefficient_upper * hi,
            ];
            total += products.into_iter().min().unwrap();
        }
        total
    }

    #[test]
    fn opposite_facets_have_strict_mixture_gain() {
        let mut bank = FacetBank::new(1).unwrap();
        bank.push(LowerAffineCertificate::new(vec![1.0], 0.0).unwrap())
            .unwrap();
        bank.push(LowerAffineCertificate::new(vec![-1.0], 0.0).unwrap())
            .unwrap();

        let result = bank.certify(&box_1d(-1.0, 1.0)).unwrap();
        assert!(result.best_one_hot <= -1.0);
        assert!(result.lower_bound > -1.0e-6);
        assert!(result.lower_bound >= result.best_one_hot);
        assert_eq!(result.weight_numerators, vec![2048, 2048]);
    }

    #[test]
    fn one_plane_identity_and_point_box() {
        let mut bank = FacetBank::new(1).unwrap();
        bank.push(LowerAffineCertificate::new(vec![3.0], 1.0).unwrap())
            .unwrap();
        let result = bank.certify(&box_1d(2.0, 2.0)).unwrap();
        assert!(result.lower_bound <= 7.0);
        assert!(result.lower_bound >= next_down_f32(7.0));
        assert_eq!(result.lower_bound, result.best_one_hot);
        assert_eq!(result.weight_numerators, vec![4096]);
    }

    #[test]
    fn coefficient_and_bias_errors_are_folded_outward() {
        let mut bank = FacetBank::new(1).unwrap();
        bank.push(LowerAffineCertificate::with_errors(vec![1.0], 0.0, vec![0.1], 0.2).unwrap())
            .unwrap();
        let result = bank.certify(&box_1d(-2.0, 3.0)).unwrap();
        assert!(result.lower_bound <= -2.4);
        assert!(result.lower_bound > -2.401);
    }

    #[test]
    fn result_never_regresses_below_best_one_hot() {
        let mut bank = FacetBank::new(2).unwrap();
        bank.push(LowerAffineCertificate::new(vec![1.0, -2.0], 0.5).unwrap())
            .unwrap();
        bank.push(LowerAffineCertificate::new(vec![-0.5, 0.25], -0.1).unwrap())
            .unwrap();
        bank.push(LowerAffineCertificate::new(vec![0.25, 1.5], 0.0).unwrap())
            .unwrap();
        let input =
            BoundedTensor::new(arr1(&[-1.0, -2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn())
                .unwrap();
        let first = bank.certify(&input).unwrap();
        let second = bank.certify(&input).unwrap();
        assert!(first.lower_bound >= first.best_one_hot);
        assert_eq!(first, second);
    }

    #[test]
    fn bad_shapes_nonfinite_data_and_bad_weights_are_rejected() {
        assert!(LowerAffineCertificate::new(vec![f32::NAN], 0.0).is_err());
        assert!(LowerAffineCertificate::new(vec![1.0], f32::INFINITY).is_err());
        assert!(LowerAffineCertificate::with_errors(vec![1.0], 0.0, vec![-0.1], 0.0).is_err());
        assert!(LowerAffineCertificate::with_errors(vec![1.0], 0.0, vec![0.1, 0.2], 0.0).is_err());

        let mut bank = FacetBank::new(1).unwrap();
        bank.push(LowerAffineCertificate::new(vec![1.0], 0.0).unwrap())
            .unwrap();
        let wrong_dim =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();
        assert!(bank.certify(&wrong_dim).is_err());
        let infinite = BoundedTensor::new_allow_infinite(
            arr1(&[f32::NEG_INFINITY]).into_dyn(),
            arr1(&[1.0]).into_dyn(),
        )
        .unwrap();
        assert!(bank.certify(&infinite).is_err());
        assert!(bank
            .certify_dyadic_mixture(&box_1d(-1.0, 1.0), &[3], 2)
            .is_err());
        assert!(bank
            .certify_dyadic_mixture(&box_1d(-1.0, 1.0), &[4, 0], 2)
            .is_err());
    }

    #[test]
    fn ninth_plane_is_rejected() {
        let mut bank = FacetBank::new(1).unwrap();
        for index in 0..FACET_BANK_MAX_PLANES {
            bank.push(LowerAffineCertificate::new(vec![index as f32], 0.0).unwrap())
                .unwrap();
        }
        assert!(bank
            .push(LowerAffineCertificate::new(vec![9.0], 0.0).unwrap())
            .is_err());
    }

    #[test]
    fn outward_interval_arithmetic_is_below_exact_rational_oracle() {
        let mut state = 0x1234_5678_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        for _case in 0..100 {
            let dimension = 1 + (next() as usize % 5);
            let plane_count = 2 + (next() as usize % 4);
            let mut planes = Vec::with_capacity(plane_count);
            for _ in 0..plane_count {
                let coefficients: Vec<f32> = (0..dimension)
                    .map(|_| ((next() % 513) as i32 - 256) as f32 / 16.0)
                    .collect();
                let errors: Vec<f32> = (0..dimension)
                    .map(|_| (next() % 9) as f32 / 1024.0)
                    .collect();
                let bias = ((next() % 257) as i32 - 128) as f32 / 32.0;
                let bias_error = (next() % 9) as f32 / 1024.0;
                planes.push(
                    LowerAffineCertificate::with_errors(coefficients, bias, errors, bias_error)
                        .unwrap(),
                );
            }
            let mut lower = Vec::with_capacity(dimension);
            let mut upper = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                let lo = -((next() % 65) as f32) / 16.0;
                let hi = ((next() % 65) as f32) / 16.0;
                lower.push(lo);
                upper.push(hi);
            }
            let input = BoundedTensor::new(
                ndarray::Array1::from_vec(lower).into_dyn(),
                ndarray::Array1::from_vec(upper).into_dyn(),
            )
            .unwrap();
            let domain = ValidatedBox::new(&input, dimension).unwrap();
            let denominator = 16_u32;
            let mut weights = vec![0_u32; plane_count];
            for _ in 0..denominator {
                weights[next() as usize % plane_count] += 1;
            }
            let certified = certify_weights_f64(&planes, &domain, &weights, denominator).unwrap();
            let exact = exact_interval_mixture_min(&planes, &domain, &weights, denominator);
            let certified_exact = BigRational::from_float(certified).unwrap();
            assert!(
                certified_exact <= exact,
                "certified={certified} exact={exact} weights={weights:?}"
            );
        }
    }
}
