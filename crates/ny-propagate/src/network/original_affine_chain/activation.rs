// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Private checked mathematical GELU-erf and SiLU scalar kernels.
//!
//! Dedicated original-affine GELU-erf and SiLU stages consume these kernels
//! under their original paid phase. This module grants no generic Layer
//! eligibility, execution contract or verifier verdict. Binary32 endpoints and the published slope
//! denote their exact real bit values. A caller retains every returned object
//! under its original phase, then attaches that phase's persistent reservation
//! to its private report owner. This module creates no budget or global cache.
//!
//! Authority comes from directed arithmetic, not host transcendental results:
//! Machin's identity checks pi; squared endpoints check sqrt proposals; the
//! existing positive-series exp checker checks exponentials. DLMF 7.6.2 gives
//! the positive erf series. The erfc tail follows by integrating
//! exp(-t*t) <= (t/z)*exp(-t*t), t >= z > 0.

use crate::certify::work::WorkRequest;
use crate::layers::linear::bias::{
    f32_to_f64_exact, f64_to_f32_down_no_subnormal, f64_to_f32_up_no_subnormal,
};
use crate::layers::linear::original_affine::{OriginalAffinePhase, OriginalAffinePhaseIdentity};
use crate::layers::softmax::weighted_support::{down, exp_nonpositive, up, Interval as I};
use ny_core::{NyError, Result};

// Arithmetic only: no imported model, graph, input-box or proof authority.
#[path = "decoder_scalar.rs"]
pub mod decoder_scalar;

const PREPARE_WORK: u64 = 32 * 1024;
const ENVELOPE_WORK: u64 = 64 * 1024;
const EXP_WORK: u64 = 4096;
const SCRATCH: usize = 32 * 1024;
const MAX_ENVELOPES: usize = 4 * 1024 * 1024;
const ARCTAN_TERMS: usize = 24;
const ERF_TERMS: usize = 96;
const POLL_WORK: u64 = 256;

fn invalid(message: &str) -> NyError {
    NyError::InvalidSpec(format!("original-affine activation: {message}"))
}
fn numeric(message: &str) -> NyError {
    NyError::NumericalInstability(format!("original-affine activation: {message}"))
}

/// Closed exact mathematical profiles. Tanh GELU and device kernels are absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MathematicalActivation {
    GeluErf,
    SiLu,
}
impl MathematicalActivation {
    #[cfg(test)]
    pub(super) fn from_declared(profile: &str, phase: &mut OriginalAffinePhase) -> Result<Self> {
        phase.entry()?;
        if profile.len() > 64 {
            return Err(invalid("profile name exceeds the closed profile limit"));
        }
        phase.charge_work(profile.len() as u64 + 1)?;
        let profile = match profile {
            "gelu-erf-exact-real/v1" => Self::GeluErf,
            "silu-exact-real/v1" => Self::SiLu,
            _ => {
                return Err(NyError::UnsupportedConfiguration(
                    "original-affine activation requires explicit GELU-erf or SiLU exact-real profile"
                        .into(),
                ));
            }
        };
        phase.checkpoint()?;
        Ok(profile)
    }
}

/// Aggregate prediction, before any scalar constants or output are constructed.
/// It counts requested capacities, not measured RSS or elapsed runtime.
/// The 4 Mi frontier concerns newly retained scalar envelopes, not model weights.
pub(super) fn request(envelopes: usize) -> Result<WorkRequest> {
    if envelopes > MAX_ENVELOPES {
        return Err(NyError::UnsupportedConfiguration(
            "original-affine activation exceeds the 4 Mi envelope profile".into(),
        ));
    }
    let work = u64::try_from(envelopes)
        .ok()
        .and_then(|n| n.checked_mul(ENVELOPE_WORK))
        .and_then(|work| work.checked_add(PREPARE_WORK))
        .ok_or_else(|| invalid("work overflow"))?;
    let persistent_bytes = envelopes
        .checked_mul(size_of::<ScalarEnvelope>())
        .and_then(|bytes| bytes.checked_add(size_of::<CheckedActivation>()))
        .and_then(|bytes| bytes.checked_add(1024))
        .ok_or_else(|| invalid("capacity overflow"))?;
    Ok(WorkRequest {
        scratch_bytes: SCRATCH,
        persistent_bytes,
        work,
        proposals: 0,
    })
}

/// Every elementary operation is prepaid. Polling is bounded by 256 credited
/// terms, except the bounded exp helper, which is polled before and after.
/// That helper performs <=18 Taylor terms and <=13 squarings on [-708,0];
/// its monotone tail branch executes that finite branch once. 4096 credits
/// conservatively cover all its arithmetic, validation and branch work.
struct Math<'a> {
    phase: &'a mut OriginalAffinePhase,
    since_poll: u64,
}
impl<'a> Math<'a> {
    fn new(phase: &'a mut OriginalAffinePhase) -> Result<Self> {
        phase.entry()?;
        Ok(Self {
            phase,
            since_poll: 0,
        })
    }
    fn pay(&mut self, amount: u64) -> Result<()> {
        self.phase.charge_work(amount)?;
        self.since_poll += amount;
        if self.since_poll >= POLL_WORK {
            self.phase.checkpoint()?;
            self.since_poll = 0;
        }
        Ok(())
    }
    fn add(&mut self, a: I, b: I) -> Result<I> {
        self.pay(8)?;
        Ok(a.add(b))
    }
    fn sub(&mut self, a: I, b: I) -> Result<I> {
        self.pay(12)?;
        Ok(a.sub(b))
    }
    fn mul(&mut self, a: I, b: I) -> Result<I> {
        self.pay(24)?;
        Ok(a.mul(b))
    }
    fn div(&mut self, a: I, b: I) -> Result<I> {
        self.pay(48)?;
        if !a.finite() || !b.finite() || b.lo <= 0.0 {
            return Err(numeric("positive finite division denominator required"));
        }
        Ok(a.div_positive(b))
    }
    fn finite(&mut self, x: I) -> Result<I> {
        self.pay(8)?;
        if !x.finite() {
            return Err(numeric("nonfinite or inverted checked interval"));
        }
        Ok(x)
    }
    fn probability(&mut self, x: I) -> Result<I> {
        let x = self.finite(x)?;
        if x.hi < 0.0 || x.lo > 1.0 {
            return Err(numeric("probability enclosure excludes [0,1]"));
        }
        // Intersect with a proved mathematical range; never repair NaN/poison.
        Ok(I::new(x.lo.max(0.0), x.hi.min(1.0)))
    }
    fn exponential(&mut self, exponent: I) -> Result<I> {
        let exponent = self.finite(exponent)?;
        if exponent.hi > 0.0 {
            return Err(numeric("exp checker requires nonpositive exponents"));
        }
        self.pay(2 * EXP_WORK)?;
        self.phase.checkpoint()?;
        let result = I::new(
            exp_nonpositive(exponent.lo).lo,
            exp_nonpositive(exponent.hi).hi,
        );
        self.phase.checkpoint()?;
        self.probability(result)
    }
    fn recurrence(&mut self) -> Result<()> {
        self.pay(4)?;
        #[cfg(test)]
        {
            RECURRENCE_CANCEL.with(|hook| {
                let mut hook = hook.borrow_mut();
                if let Some((remaining, token)) = hook.as_mut() {
                    *remaining -= 1;
                    if *remaining == 0 {
                        token.cancel();
                        *hook = None;
                    }
                }
            });
        }
        self.phase.checkpoint()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EnvelopeKind {
    Taylor,
    /// The global x*p(x), p in [0,1], hull when finite f32 Taylor publication
    /// would overflow. It still encloses the same explicit real activation.
    GlobalRange,
}

/// Constructors and fields are private. The lower/upper lines have the SAME
/// exact stored slope; every slope-storage residual is discharged into bias.
/// Images are outward binary32 normal/zero endpoints, not runtime predictions.
#[derive(Debug)]
pub(super) struct ScalarEnvelope {
    profile: MathematicalActivation,
    lower: f32,
    upper: f32,
    center: f32,
    slope: f32,
    lower_intercept: f32,
    upper_intercept: f32,
    image_lower: f32,
    image_upper: f32,
    kind: EnvelopeKind,
}
impl ScalarEnvelope {
    pub(super) fn slope(&self) -> f32 {
        self.slope
    }
    pub(super) fn intercepts(&self) -> (f32, f32) {
        (self.lower_intercept, self.upper_intercept)
    }
    pub(super) fn image(&self) -> (f32, f32) {
        (self.image_lower, self.image_upper)
    }
    pub(super) fn profile(&self) -> MathematicalActivation {
        self.profile
    }
    pub(super) fn interval(&self) -> (f32, f32) {
        (self.lower, self.upper)
    }
    pub(super) fn kind(&self) -> EnvelopeKind {
        self.kind
    }

    /// Recheck the construction metadata when a paid fact is consumed. This
    /// checks the stored center and the closed global-hull representation;
    /// it does not reconstruct a Taylor proof or authenticate an input box.
    pub(super) fn validate_construction_metadata(
        &self,
        phase: &mut OriginalAffinePhase,
    ) -> Result<()> {
        phase.charge_work(32)?;
        let lower = f32_to_f64_exact(self.lower);
        let upper = f32_to_f64_exact(self.upper);
        let center = f32_to_f64_exact(self.center);
        if !center.is_finite() || center < lower || center > upper {
            return Err(numeric("checked activation center is outside its interval"));
        }
        if self.kind() == EnvelopeKind::GlobalRange {
            let lower = f64_to_f32_down_no_subnormal(lower.min(0.0));
            let upper = f64_to_f32_up_no_subnormal(upper.max(0.0));
            if self.slope != 0.0
                || self.lower_intercept.to_bits() != lower.to_bits()
                || self.upper_intercept.to_bits() != upper.to_bits()
                || self.image_lower.to_bits() != lower.to_bits()
                || self.image_upper.to_bits() != upper.to_bits()
            {
                return Err(numeric("checked activation global hull metadata mismatch"));
            }
        }
        phase.checkpoint()
    }
}

/// Immutable checked constants are built anew under the caller's finite phase.
/// No host pi, erf, logistic or exp value can enter this constructor.
#[derive(Debug)]
pub(super) struct CheckedActivation {
    phase_identity: OriginalAffinePhaseIdentity,
    profile: MathematicalActivation,
    pi: I,
    root_two: I,
    root_pi: I,
    inverse_root_two_pi: I,
}
impl CheckedActivation {
    pub(super) fn prepare(
        profile: MathematicalActivation,
        phase: &mut OriginalAffinePhase,
    ) -> Result<Self> {
        phase.entry()?;
        let phase_identity = phase.identity();
        phase.retain_metadata(size_of::<Self>())?;
        let mut math = Math::new(phase)?;
        let one_fifth = math.div(I::point(1.0), I::point(5.0))?;
        let one_239 = math.div(I::point(1.0), I::point(239.0))?;
        let a = arctan_small(one_fifth, &mut math)?;
        let b = arctan_small(one_239, &mut math)?;
        let a = math.mul(I::point(16.0), a)?;
        let b = math.mul(I::point(4.0), b)?;
        let pi = math.sub(a, b)?;
        let pi = math.finite(pi)?;
        if pi.lo <= 2.0 || pi.hi >= 4.0 {
            return Err(numeric("Machin pi enclosure failed its range check"));
        }
        let root_two = checked_sqrt(I::point(2.0), &mut math)?;
        let root_pi = checked_sqrt(pi, &mut math)?;
        let two_pi = math.mul(I::point(2.0), pi)?;
        let root_two_pi = checked_sqrt(two_pi, &mut math)?;
        let inverse_root_two_pi = math.div(I::point(1.0), root_two_pi)?;
        // This directed check also proves the GELU global curvature bound
        // 2/sqrt(2*pi) is strictly below the earlier M=1 profile.
        if inverse_root_two_pi.hi >= 0.5 {
            return Err(numeric(
                "normal-density constant exceeds proved curvature profile",
            ));
        }
        math.phase.checkpoint()?;
        Ok(Self {
            phase_identity,
            profile,
            pi,
            root_two,
            root_pi,
            inverse_root_two_pi,
        })
    }

    /// No work/retention changes on a foreign phase, even on the same ledger.
    /// Constants may be retained in their private report, but their arithmetic
    /// cannot silently become free preparation for a separately admitted phase.
    fn require_phase(&self, phase: &OriginalAffinePhase) -> Result<()> {
        if self.phase_identity != phase.identity() {
            return Err(NyError::UnsupportedConfiguration(
                "original-affine activation preparation belongs to another phase".into(),
            ));
        }
        Ok(())
    }

    fn validate(&self, math: &mut Math<'_>) -> Result<()> {
        self.require_phase(math.phase)?;
        for x in [
            self.pi,
            self.root_two,
            self.root_pi,
            self.inverse_root_two_pi,
        ] {
            math.finite(x)?;
            if x.lo <= 0.0 {
                return Err(numeric("poisoned checked activation constants"));
            }
        }
        if self.pi.lo <= 2.0 || self.pi.hi >= 4.0 || self.inverse_root_two_pi.hi >= 0.5 {
            return Err(numeric("poisoned curvature profile"));
        }
        Ok(())
    }

    pub(super) fn enclose(
        &self,
        lower: f32,
        upper: f32,
        phase: &mut OriginalAffinePhase,
    ) -> Result<ScalarEnvelope> {
        self.enclose_with_slope(lower, upper, None, phase)
    }

    fn enclose_with_slope(
        &self,
        lower: f32,
        upper: f32,
        proposed_slope: Option<f32>,
        phase: &mut OriginalAffinePhase,
    ) -> Result<ScalarEnvelope> {
        self.require_phase(phase)?;
        phase.entry()?;
        phase.charge_work(16)?;
        // Bit decoding retains original binary32 subnormals even under DAZ.
        let l = f32_to_f64_exact(lower);
        let u = f32_to_f64_exact(upper);
        if !l.is_finite() || !u.is_finite() || l > u {
            return Err(invalid("finite ordered scalar endpoints required"));
        }
        if proposed_slope.is_some_and(|a| !f32_to_f64_exact(a).is_finite()) {
            return Err(numeric("nonfinite stored slope proposal"));
        }
        phase.retain_metadata(size_of::<ScalarEnvelope>())?;
        let mut math = Math::new(phase)?;
        self.validate(&mut math)?;
        // Host midpoint/cast proposes a center. Membership is checked using
        // exact bit interpretation, so rounded/flushed proposals grant nothing.
        math.pay(16)?;
        let proposal = (0.5 * l + 0.5 * u) as f32;
        let p = f32_to_f64_exact(proposal);
        let center = if p.is_finite() && l <= p && p <= u {
            proposal
        } else {
            lower
        };
        let c = f32_to_f64_exact(center);
        let (value, derivative) = self.value_derivative(c, &mut math)?;
        math.pay(16)?;
        let mut slope = proposed_slope.unwrap_or_else(|| derivative.midpoint() as f32);
        if slope.to_bits() & 0x7f80_0000 == 0 {
            // Intentional normal/zero storage; its full loss is paid below.
            slope = 0.0;
        }
        let a = f32_to_f64_exact(slope);
        if !a.is_finite() {
            return Err(numeric("stored slope overflow"));
        }
        let left = math.sub(I::point(l), I::point(c))?;
        let right = math.sub(I::point(u), I::point(c))?;
        math.pay(8)?;
        let radius = left.abs_max().max(right.abs_max());
        let slope_error = math.sub(derivative, I::point(a))?;
        let slope_error = math.mul(I::point(radius), I::point(slope_error.abs_max()))?;
        let radius_squared = math.mul(I::point(radius), I::point(radius))?;
        math.pay(8)?;
        let half_curvature = self.half_global_curvature();
        let curvature_error = math.mul(half_curvature, radius_squared)?;
        let error = math.add(slope_error, curvature_error)?;
        let error = math.finite(error)?;
        let a_center = math.mul(I::point(a), I::point(c))?;
        let intercept = math.sub(value, a_center)?;
        let padded = math.add(intercept, I::new(-error.hi, error.hi))?;
        let padded = math.finite(padded)?;
        math.pay(16)?;
        let low_b = f64_to_f32_down_no_subnormal(padded.lo);
        let high_b = f64_to_f32_up_no_subnormal(padded.hi);
        let global = I::new(l.min(0.0), u.max(0.0));
        let (slope, low_b, high_b, image, kind) = if low_b.is_finite() && high_b.is_finite() {
            let ax = math.mul(I::point(a), I::new(l, u))?;
            let image = math.add(
                ax,
                I::new(f32_to_f64_exact(low_b), f32_to_f64_exact(high_b)),
            )?;
            let image = math.finite(image)?;
            // Both sets enclose the same function. Intersection is valid even
            // if the Taylor lines are too broad to improve the global range.
            let image = I::new(image.lo.max(global.lo), image.hi.min(global.hi));
            let image = math.finite(image)?;
            (slope, low_b, high_b, image, EnvelopeKind::Taylor)
        } else {
            math.pay(16)?;
            let low_b = f64_to_f32_down_no_subnormal(global.lo);
            let high_b = f64_to_f32_up_no_subnormal(global.hi);
            (0.0, low_b, high_b, global, EnvelopeKind::GlobalRange)
        };
        math.pay(32)?;
        let image_lower = f64_to_f32_down_no_subnormal(image.lo);
        let image_upper = f64_to_f32_up_no_subnormal(image.hi);
        if !image_lower.is_finite() || !image_upper.is_finite() || image_lower > image_upper {
            return Err(numeric("scalar image publication failed"));
        }
        math.phase.checkpoint()?;
        Ok(ScalarEnvelope {
            profile: self.profile,
            lower,
            upper,
            center,
            slope,
            lower_intercept: low_b,
            upper_intercept: high_b,
            image_lower,
            image_upper,
            kind,
        })
    }

    // Taylor remainder is M*r^2/2. These are analytic global bounds, not
    // sampled extrema or interval-specific curvature proposals.
    fn half_global_curvature(&self) -> I {
        match self.profile {
            MathematicalActivation::GeluErf => {
                // f''(x)=(2-x^2)*exp(-x^2/2)/sqrt(2*pi). For z=x^2<=2,
                // |2-z|*exp(-z/2)<=2. For z>=2, exp(z/2)>=z/2 implies
                // (z-2)*exp(-z/2)<=2*(z-2)/z<=2. Thus M/2 is bounded
                // by the already checked inverse normal-density constant.
                self.inverse_root_two_pi
            }
            MathematicalActivation::SiLu => {
                // With y=|x|/2, f''=(1-y*tanh(y))/(2*cosh(y)^2).
                // A nonnegative numerator is <=1. Otherwise its magnitude
                // is <=y-1<=1+y^2<=cosh(y)^2, since sinh(y)>=y.
                // Therefore |f''|<=1/2 and its half is exactly 1/4.
                I::point(0.25)
            }
        }
    }

    fn value_derivative(&self, x: f64, math: &mut Math<'_>) -> Result<(I, I)> {
        self.require_phase(math.phase)?;
        math.pay(16)?;
        if !x.is_finite() {
            return Err(numeric("finite scalar evaluation point required"));
        }
        if x == 0.0 {
            return Ok((I::point(0.0), I::point(0.5)));
        }
        match self.profile {
            MathematicalActivation::GeluErf => {
                let z = math.div(I::point(x.abs()), self.root_two)?;
                let tail = self.erfc(z, math)?;
                let half_tail = math.mul(I::point(0.5), tail)?;
                let probability = if x > 0.0 {
                    math.sub(I::point(1.0), half_tail)?
                } else {
                    half_tail
                };
                let probability = math.probability(probability)?;
                // Negative tail multiplication avoids subtracting nearly equal
                // probabilities before multiplying by a negative input.
                let value = math.mul(I::point(x), probability)?;
                let square = math.mul(I::point(x), I::point(x))?;
                let half_square = math.mul(I::point(-0.5), square)?;
                let exponential = math.exponential(half_square)?;
                let density = math.mul(exponential, self.inverse_root_two_pi)?;
                let x_density = math.mul(I::point(x), density)?;
                let derivative = math.add(probability, x_density)?;
                Ok((math.finite(value)?, math.finite(derivative)?))
            }
            MathematicalActivation::SiLu => {
                let exponential = math.exponential(I::point(-x.abs()))?;
                let denominator = math.add(I::point(1.0), exponential)?;
                let numerator = if x > 0.0 { I::point(1.0) } else { exponential };
                let probability = math.div(numerator, denominator)?;
                let probability = math.probability(probability)?;
                let value = math.mul(I::point(x), probability)?;
                // Reuse the same checked e=exp(-|x|). For positive large x,
                // subtraction from a rounded probability can lose the small
                // complement. Its direct quotient preserves that tail and
                // pays this division on the caller's original phase.
                let complement_numerator = if x > 0.0 { exponential } else { I::point(1.0) };
                let complement = math.div(complement_numerator, denominator)?;
                let complement = math.probability(complement)?;
                let product = math.mul(probability, complement)?;
                let x_product = math.mul(I::point(x), product)?;
                let derivative = math.add(probability, x_product)?;
                Ok((math.finite(value)?, math.finite(derivative)?))
            }
        }
    }

    fn erfc(&self, z: I, math: &mut Math<'_>) -> Result<I> {
        self.require_phase(math.phase)?;
        let z = math.finite(z)?;
        if z.lo < 0.0 {
            return Err(numeric("erfc argument must be nonnegative"));
        }
        // erfc is decreasing. Only independently enclosed exact endpoint
        // functions authorize this range, including a branch-straddling z.
        let lower = self.erfc_point(z.hi, math)?;
        let upper = self.erfc_point(z.lo, math)?;
        math.probability(I::new(lower.lo, upper.hi))
    }

    fn erfc_point(&self, z: f64, math: &mut Math<'_>) -> Result<I> {
        self.require_phase(math.phase)?;
        math.pay(8)?;
        if !z.is_finite() || z < 0.0 {
            return Err(numeric("finite nonnegative erfc point required"));
        }
        if z == 0.0 {
            return Ok(I::point(1.0));
        }
        let square = math.mul(I::point(z), I::point(z))?;
        let exponential = math.exponential(square.neg())?;
        if z > 4.0 {
            let denominator = math.mul(I::point(z), self.root_pi)?;
            let tail = math.div(exponential, denominator)?;
            return math.probability(I::new(0.0, tail.hi));
        }
        // DLMF 7.6.2: erf(z)=(2/sqrt(pi))*exp(-z^2)*sum(t_n),
        // t_0=z, t_(n+1)=t_n*2*z^2/(2*n+3). All terms are positive.
        let twice_square = math.mul(I::point(2.0), square)?;
        let mut term = I::point(z);
        let mut sum = term;
        for n in 0..ERF_TERMS - 1 {
            math.recurrence()?;
            let product = math.mul(term, twice_square)?;
            term = math.div(product, I::point((2 * n + 3) as f64))?;
            sum = math.add(sum, term)?;
        }
        let product = math.mul(term, twice_square)?;
        let next = math.div(product, I::point((2 * ERF_TERMS + 1) as f64))?;
        let ratio = math.div(twice_square, I::point((2 * ERF_TERMS + 3) as f64))?;
        if ratio.hi >= 1.0 {
            return Err(numeric("erf geometric remainder ratio failed"));
        }
        let denominator = math.sub(I::point(1.0), I::point(ratio.hi))?;
        let remainder = math.div(I::new(0.0, next.hi), denominator)?;
        sum = math.add(sum, remainder)?;
        let normalization = math.div(I::point(2.0), self.root_pi)?;
        let scaled = math.mul(normalization, exponential)?;
        let erf = math.mul(scaled, sum)?;
        let erf = math.probability(erf)?;
        let tail = math.sub(I::point(1.0), erf)?;
        math.probability(tail)
    }
}

/// Atan's alternating first-omitted-term bound is valid for 0 <= x < 1.
/// Endpoint uncertainty is propagated through every finite polynomial term.
fn arctan_small(x: I, math: &mut Math<'_>) -> Result<I> {
    let x = math.finite(x)?;
    if x.lo < 0.0 || x.hi >= 1.0 {
        return Err(numeric("small arctan profile required"));
    }
    let square = math.mul(x, x)?;
    let mut power = x;
    let mut sum = I::point(0.0);
    for n in 0..ARCTAN_TERMS {
        math.recurrence()?;
        let term = math.div(power, I::point((2 * n + 1) as f64))?;
        sum = if n % 2 == 0 {
            math.add(sum, term)?
        } else {
            math.sub(sum, term)?
        };
        power = math.mul(power, square)?;
    }
    let remainder = math.div(power, I::point((2 * ARCTAN_TERMS + 1) as f64))?;
    // ARCTAN_TERMS is even: the first omitted term is nonnegative.
    let upper_tail = I::new(0.0, remainder.hi);
    math.add(sum, upper_tail)
}

/// The host supplies ONLY a proposal. Directed squaring checks inequalities
/// against the entire input interval; bounded widening either checks or refuses.
fn sqrt_square_certified(
    proposal: f64,
    argument: f64,
    lower: bool,
    math: &mut Math<'_>,
) -> Result<bool> {
    // Squaring a tiny root directly can round every adjacent proposal to the
    // same subnormal. Normalize each endpoint independently by exact powers
    // of two, then certify the scaled inequality with outward arithmetic.
    let large = f64::from_bits((1023u64 + 512) << 52);
    let small = f64::from_bits((1023u64 - 512) << 52);
    let scale = if argument < small {
        large
    } else if argument > large {
        small
    } else {
        1.0
    };
    let (root, input) = if scale == 1.0 {
        (I::point(proposal), I::point(argument))
    } else {
        let scale = I::point(scale);
        let root = math.mul(I::point(proposal), scale)?;
        let input = math.mul(I::point(argument), scale)?;
        (root, math.mul(input, scale)?)
    };
    let square = math.mul(root, root)?;
    Ok(if lower {
        square.hi <= input.lo
    } else {
        square.lo >= input.hi
    })
}

fn checked_sqrt(x: I, math: &mut Math<'_>) -> Result<I> {
    let x = math.finite(x)?;
    if x.lo < 0.0 {
        return Err(numeric("nonnegative sqrt profile required"));
    }
    if x.hi == 0.0 {
        return Ok(I::point(0.0));
    }
    math.pay(32)?;
    let mut lower = down(x.lo.sqrt()).max(0.0);
    let mut upper = up(x.hi.sqrt());
    for _ in 0..8 {
        math.recurrence()?;
        let lower_certified = lower == 0.0 || sqrt_square_certified(lower, x.lo, true, math)?;
        let upper_certified = sqrt_square_certified(upper, x.hi, false, math)?;
        if lower_certified && upper_certified {
            return Ok(I::new(lower, upper));
        }
        math.pay(8)?;
        lower = down(lower).max(0.0);
        upper = up(upper);
    }
    Err(numeric("sqrt proposal did not pass directed square checks"))
}

// Taylor authority: |f''| <= 1 globally for BOTH explicit profiles.
// GELU: f''=phi(x)*(2-x^2). With t=x^2 <=2, |2-t|*exp(-t/2)<=2.
// For t>2, (t-2)*exp(-t/2) has maximum 2*exp(-2) at t=4, also <=2.
// Checked pi>2 gives 1/sqrt(2*pi)<1/2; hence |f''|<1.
// SiLU: f''=s(1-s)*(2+x*(1-2*s)). s(1-s)<=1/4 and <=exp(-|x|),
// so |f''|<=1/2+|x|*exp(-|x|)<=1/2+1/e<1 (e>1+1=2).
// For any stored finite a, f(x)-a*x lies in
// F(c)-a*c +/- [r*max|D(c)-a|+r^2/2]. Every term is rounded outward.

#[cfg(test)]
thread_local! {
    static RECURRENCE_CANCEL: std::cell::RefCell<Option<(usize, crate::certify::residency::CancellationToken)>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certify::residency::{CancellationToken, MemoryAuthority};
    use crate::certify::work::{LeaseClass, WorkBudget, WorkLimits, WorkReserve};
    use num_bigint::BigInt;
    use num_rational::BigRational as Q;
    use num_traits::{One, Signed, Zero};
    use std::time::{Duration, Instant};

    fn admitted(req: WorkRequest) -> (WorkBudget, OriginalAffinePhase) {
        let budget = WorkBudget::new(
            None,
            req.scratch_bytes,
            req.persistent_bytes,
            req.work,
            req.proposals,
            Duration::ZERO,
        )
        .unwrap();
        let phase = OriginalAffinePhase::admit(&budget, LeaseClass::Optional, req, None).unwrap();
        (budget, phase)
    }
    fn prepared(
        profile: MathematicalActivation,
        envelopes: usize,
    ) -> (WorkBudget, OriginalAffinePhase, CheckedActivation) {
        let (budget, mut phase) = admitted(request(envelopes).unwrap());
        let kernel = CheckedActivation::prepare(profile, &mut phase).unwrap();
        (budget, phase, kernel)
    }
    fn integer(n: i64) -> Q {
        Q::from_integer(BigInt::from(n))
    }
    fn ratio(n: i64, d: i64) -> Q {
        Q::new(BigInt::from(n), BigInt::from(d))
    }

    // Independent bit interpretation. No kernel conversion/interval/libm
    // implementation is used by these exact rational oracles.
    fn dyadic(sign: bool, mantissa: u64, power: i32) -> Q {
        let mut n = BigInt::from(mantissa);
        if sign {
            n = -n;
        }
        if power >= 0 {
            Q::from_integer(n << power as usize)
        } else {
            Q::new(n, BigInt::from(1) << (-power) as usize)
        }
    }
    fn q32(value: f32) -> Q {
        let bits = value.to_bits();
        let exponent = (bits >> 23) & 255;
        assert_ne!(exponent, 255);
        let fraction = bits & 0x7f_ffff;
        let (mantissa, power) = if exponent == 0 {
            (u64::from(fraction), -149)
        } else {
            (u64::from(fraction | (1 << 23)), exponent as i32 - 127 - 23)
        };
        dyadic(bits >> 31 != 0, mantissa, power)
    }
    fn q64(value: f64) -> Q {
        let bits = value.to_bits();
        let exponent = (bits >> 52) & 2047;
        assert_ne!(exponent, 2047);
        let fraction = bits & 0x000f_ffff_ffff_ffff;
        let (mantissa, power) = if exponent == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1 << 52), exponent as i32 - 1023 - 52)
        };
        dyadic(bits >> 63 != 0, mantissa, power)
    }
    #[derive(Debug, Clone)]
    struct RationalInterval {
        lo: Q,
        hi: Q,
    }
    impl RationalInterval {
        fn point(x: Q) -> Self {
            Self {
                lo: x.clone(),
                hi: x,
            }
        }
        fn add(&self, rhs: &Self) -> Self {
            Self {
                lo: &self.lo + &rhs.lo,
                hi: &self.hi + &rhs.hi,
            }
        }
        fn sub(&self, rhs: &Self) -> Self {
            Self {
                lo: &self.lo - &rhs.hi,
                hi: &self.hi - &rhs.lo,
            }
        }
        fn mul(&self, rhs: &Self) -> Self {
            let values = [
                &self.lo * &rhs.lo,
                &self.lo * &rhs.hi,
                &self.hi * &rhs.lo,
                &self.hi * &rhs.hi,
            ];
            Self {
                lo: values.iter().min().unwrap().clone(),
                hi: values.iter().max().unwrap().clone(),
            }
        }
        fn div(&self, rhs: &Self) -> Self {
            assert!(rhs.lo > Q::zero());
            self.mul(&Self {
                lo: rhs.hi.recip(),
                hi: rhs.lo.recip(),
            })
        }
    }
    fn atan_rational(x: Q) -> RationalInterval {
        assert!(x >= Q::zero() && x < Q::one());
        let square = &x * &x;
        let mut power = x;
        let mut sum = Q::zero();
        // Different precision and all-exact arithmetic from the checked kernel.
        for n in 0..64 {
            let term = &power / integer(2 * n + 1);
            if n % 2 == 0 {
                sum += term
            } else {
                sum -= term
            }
            power *= &square;
        }
        RationalInterval {
            lo: sum.clone(),
            hi: sum + power / integer(129),
        }
    }
    fn pi_rational() -> RationalInterval {
        atan_rational(ratio(1, 5))
            .mul(&RationalInterval::point(integer(16)))
            .sub(&atan_rational(ratio(1, 239)).mul(&RationalInterval::point(integer(4))))
    }
    fn sqrt_rational(x: &RationalInterval) -> RationalInterval {
        // Pure rational bisection, no floating sqrt proposals.
        let mut lo = Q::zero();
        let mut hi = integer(4);
        assert!(&hi * &hi >= x.hi);
        for _ in 0..96 {
            let mid = (&lo + &hi) / integer(2);
            if &mid * &mid <= x.lo {
                lo = mid
            } else {
                hi = mid
            }
        }
        let lower = lo;
        let mut lo = Q::zero();
        let mut hi = integer(4);
        for _ in 0..96 {
            let mid = (&lo + &hi) / integer(2);
            if &mid * &mid >= x.hi {
                hi = mid
            } else {
                lo = mid
            }
        }
        assert!(&lower * &lower <= x.lo);
        assert!(&hi * &hi >= x.hi);
        RationalInterval { lo: lower, hi }
    }
    fn exp_negative_rational(x: Q) -> RationalInterval {
        assert!(x >= Q::zero() && x <= integer(40));
        let mut term = Q::one();
        let mut sum = term.clone();
        // exp(+x), positive exact series then reciprocal. No production range
        // reduction, exp checker or host transcendental operation is reused.
        for n in 1..=128 {
            term = term * &x / integer(n);
            sum += &term;
        }
        let next = term * &x / integer(129);
        let future_ratio = &x / integer(130);
        assert!(future_ratio < Q::one());
        let tail = next / (Q::one() - future_ratio);
        RationalInterval {
            lo: (&sum + tail).recip(),
            hi: sum.recip(),
        }
    }
    struct Oracle {
        root_pi: RationalInterval,
        inverse_root_two_pi: RationalInterval,
    }
    impl Oracle {
        fn new() -> Self {
            let pi = pi_rational();
            let root_pi = sqrt_rational(&pi);
            let two_pi = pi.mul(&RationalInterval::point(integer(2)));
            let root = sqrt_rational(&two_pi);
            let inverse_root_two_pi = RationalInterval::point(Q::one()).div(&root);
            Self {
                root_pi,
                inverse_root_two_pi,
            }
        }
        fn erf(&self, z: Q) -> RationalInterval {
            assert!(z >= Q::zero() && z <= integer(4));
            let square = &z * &z;
            let exponential = exp_negative_rational(square.clone());
            let mut term = z;
            let mut sum = term.clone();
            // This independently sums 64 terms, with a different remainder.
            for n in 0..63 {
                term = term * integer(2) * &square / integer(2 * n + 3);
                sum += &term;
            }
            let next = term * integer(2) * &square / integer(129);
            let future_ratio = integer(2) * &square / integer(131);
            assert!(future_ratio < Q::one());
            let tail = next / (Q::one() - future_ratio);
            let sum = RationalInterval {
                lo: sum.clone(),
                hi: sum + tail,
            };
            RationalInterval::point(integer(2))
                .div(&self.root_pi)
                .mul(&exponential)
                .mul(&sum)
        }
        fn normal_probability(&self, x: Q) -> RationalInterval {
            let t = x.abs();
            let square = &t * &t;
            let exponential = exp_negative_rational(&square / integer(2));
            let density = exponential.mul(&self.inverse_root_two_pi);
            let positive = if t <= integer(4) {
                // Integrating the normal density gives the positive series in
                // x directly; unlike the kernel this never forms x/sqrt(2).
                let mut term = t;
                let mut sum = term.clone();
                for n in 0..63 {
                    term = term * &square / integer(2 * n + 3);
                    sum += &term;
                }
                let next = term * &square / integer(129);
                let future_ratio = &square / integer(131);
                let tail = next / (Q::one() - future_ratio);
                RationalInterval::point(ratio(1, 2)).add(&density.mul(&RationalInterval {
                    lo: sum.clone(),
                    hi: sum + tail,
                }))
            } else {
                // Independent integral lower/upper Mills bounds, not the
                // kernel's erfc upper-only interval. This covers x<=8 here.
                let tail = RationalInterval {
                    lo: &density.lo * &t / (&square + Q::one()),
                    hi: &density.hi / &t,
                };
                RationalInterval::point(Q::one()).sub(&tail)
            };
            if x < Q::zero() {
                RationalInterval::point(Q::one()).sub(&positive)
            } else {
                positive
            }
        }
        fn gelu(&self, x: Q) -> RationalInterval {
            RationalInterval::point(x.clone()).mul(&self.normal_probability(x))
        }
        fn silu(&self, x: Q) -> RationalInterval {
            let e = exp_negative_rational(x.abs());
            let denom = RationalInterval::point(Q::one()).add(&e);
            let probability = if x >= Q::zero() {
                RationalInterval::point(Q::one()).div(&denom)
            } else {
                e.div(&denom)
            };
            RationalInterval::point(x).mul(&probability)
        }
        fn silu_derivative(&self, x: Q) -> RationalInterval {
            let e = exp_negative_rational(x.abs());
            let denom = RationalInterval::point(Q::one()).add(&e);
            // An independent rational quotient for the original derivative,
            // rather than the production probability/complement product.
            let numerator = if x >= Q::zero() {
                denom.add(&RationalInterval::point(x).mul(&e))
            } else {
                e.mul(&denom.add(&RationalInterval::point(x)))
            };
            numerator.div(&denom.mul(&denom))
        }
        fn value(&self, profile: MathematicalActivation, x: Q) -> RationalInterval {
            match profile {
                MathematicalActivation::GeluErf => self.gelu(x),
                MathematicalActivation::SiLu => self.silu(x),
            }
        }
    }
    fn contains(checked: I, oracle: &RationalInterval) {
        assert!(
            q64(checked.lo) <= oracle.lo,
            "checked lower misses rational oracle"
        );
        assert!(
            q64(checked.hi) >= oracle.hi,
            "checked upper misses rational oracle"
        );
    }
    fn contains_envelope(envelope: &ScalarEnvelope, x: Q, value: &RationalInterval) {
        assert!(q32(envelope.lower) <= x && x <= q32(envelope.upper));
        let line = q32(envelope.slope) * x;
        assert!(&line + q32(envelope.lower_intercept) <= value.lo);
        assert!(&line + q32(envelope.upper_intercept) >= value.hi);
        assert!(q32(envelope.image_lower) <= value.lo);
        assert!(q32(envelope.image_upper) >= value.hi);
    }

    #[test]
    fn original_affine_activation_constants_have_independent_rational_authority() {
        let (_, mut phase, kernel) = prepared(MathematicalActivation::GeluErf, 0);
        let pi = pi_rational();
        contains(kernel.pi, &pi);
        assert!(q64(kernel.pi.lo) > integer(2));
        for (root, source) in [
            (kernel.root_two, RationalInterval::point(integer(2))),
            (kernel.root_pi, pi),
        ] {
            assert!(q64(root.lo) * q64(root.lo) <= source.lo);
            assert!(q64(root.hi) * q64(root.hi) >= source.hi);
        }
        let oracle = Oracle::new();
        contains(kernel.inverse_root_two_pi, &oracle.inverse_root_two_pi);
        // A sqrt proposal is never admitted without square inequalities.
        let mut math = Math::new(&mut phase).unwrap();
        assert!(checked_sqrt(I::new(-1.0, 2.0), &mut math).is_err());
        assert!(checked_sqrt(I::new(f64::NAN, 2.0), &mut math).is_err());
    }

    #[test]
    fn original_affine_activation_erf_positive_series_and_branch_tail_rational() {
        let (_, mut phase, kernel) = prepared(MathematicalActivation::GeluErf, 4);
        let oracle = Oracle::new();
        for z in [0.5, 1.0, 2.0, 4.0] {
            let mut math = Math::new(&mut phase).unwrap();
            let tail = kernel.erfc_point(z, &mut math).unwrap();
            let erf = I::point(1.0).sub(tail);
            let expected = oracle.erf(q64(z));
            contains(erf, &expected);
            assert!(tail.lo >= 0.0 && tail.hi <= 1.0);
        }
        // An argument enclosure straddling the series/tail boundary must
        // combine both checked endpoint branches, rather than selecting one
        // branch from a host midpoint or erasing the tail as exact zero.
        let mut math = Math::new(&mut phase).unwrap();
        let crossing = kernel.erfc(I::new(down(4.0), up(4.0)), &mut math).unwrap();
        let expected = RationalInterval::point(Q::one()).sub(&oracle.erf(integer(4)));
        contains(crossing, &expected);
        assert!(crossing.lo == 0.0 && crossing.hi > 0.0);

        // The >4 branch is explicit, retains a nonzero upper endpoint, and
        // includes an independently bounded negative Gaussian tail value.
        let mut math = Math::new(&mut phase).unwrap();
        let (value, _) = kernel.value_derivative(-8.0, &mut math).unwrap();
        contains(value, &oracle.gelu(integer(-8)));
        assert!(value.lo < 0.0 && value.hi < -value.lo);
    }

    #[test]
    fn original_affine_activation_taylor_lines_and_images_match_rational_oracles() {
        let oracle = Oracle::new();
        for profile in [
            MathematicalActivation::GeluErf,
            MathematicalActivation::SiLu,
        ] {
            let (_, mut phase, kernel) = prepared(profile, 3);
            for (l, u) in [(-1.5, -0.5), (-0.5, 0.75), (0.5, 1.5)] {
                let enclosure = kernel.enclose(l, u, &mut phase).unwrap();
                assert_eq!(enclosure.profile(), profile);
                assert_eq!(enclosure.interval(), (l, u));
                assert_eq!(enclosure.kind(), EnvelopeKind::Taylor);
                for x in [l, f32::midpoint(l, u), u] {
                    let x = q32(x);
                    contains_envelope(&enclosure, x.clone(), &oracle.value(profile, x));
                }
                assert!(enclosure.image_lower.is_finite() && enclosure.image_upper.is_finite());
            }
        }
    }

    #[test]
    fn original_affine_activation_tighter_global_curvature_contains_both_signs_and_tails() {
        let oracle = Oracle::new();
        for profile in [
            MathematicalActivation::GeluErf,
            MathematicalActivation::SiLu,
        ] {
            let req = request(5).unwrap();
            let (budget, mut phase, kernel) = prepared(profile, 5);
            match profile {
                MathematicalActivation::GeluErf => {
                    contains(kernel.half_global_curvature(), &oracle.inverse_root_two_pi);
                    assert!(kernel.half_global_curvature().hi < 0.5);
                }
                MathematicalActivation::SiLu => {
                    assert_eq!(q64(kernel.half_global_curvature().lo), ratio(1, 4));
                    assert_eq!(q64(kernel.half_global_curvature().hi), ratio(1, 4));
                }
            }
            for (l, u) in [
                (-8.0, -7.0),
                (-1.5, -0.5),
                (-0.5, 0.5),
                (0.5, 1.5),
                (7.0, 8.0),
            ] {
                let envelope = kernel.enclose(l, u, &mut phase).unwrap();
                for x in [l, f32::midpoint(l, u), u] {
                    let x = q32(x);
                    contains_envelope(&envelope, x.clone(), &oracle.value(profile, x));
                }
            }
            // The original admitted finite census/capacities are unchanged;
            // the selected multiplier and extra quotient spend these credits.
            assert_eq!(budget.receipt().unwrap().admitted, 1);
            assert_eq!(budget.receipt().unwrap().work_reserved, req.work);
            assert_eq!(
                budget.receipt().unwrap().live_persistent_bytes,
                req.persistent_bytes
            );
        }
    }

    #[test]
    fn original_affine_activation_tighter_curvature_strictly_narrows_old_m1_intercepts() {
        let oracle = Oracle::new();
        for profile in [
            MathematicalActivation::GeluErf,
            MathematicalActivation::SiLu,
        ] {
            let (_, mut phase, kernel) = prepared(profile, 1);
            let envelope = kernel.enclose(-0.5, 0.5, &mut phase).unwrap();
            assert_eq!(envelope.kind(), EnvelopeKind::Taylor);
            assert_eq!(q32(envelope.center), Q::zero());
            assert_eq!(q32(envelope.slope), ratio(1, 2));
            // Both original profiles have exact f(0)=0, f'(0)=1/2. With the
            // same center/slope/radius, old M=1 gives intercepts +/-1/8
            // before outward publication, hence a width of at least 1/4.
            assert!(q32(envelope.lower_intercept) > -ratio(1, 8));
            assert!(q32(envelope.upper_intercept) < ratio(1, 8));
            assert!(q32(envelope.upper_intercept) - q32(envelope.lower_intercept) < ratio(1, 4));
            for x in [
                ratio(-1, 2),
                ratio(-1, 4),
                Q::zero(),
                ratio(1, 4),
                ratio(1, 2),
            ] {
                contains_envelope(&envelope, x.clone(), &oracle.value(profile, x));
            }
        }
    }

    #[test]
    fn original_affine_activation_silu_direct_complement_retains_large_point_derivative() {
        let oracle = Oracle::new();
        let (budget, mut phase, kernel) = prepared(MathematicalActivation::SiLu, 4);
        for x in [-32.0, 32.0] {
            let (value, derivative) = {
                let mut math = Math::new(&mut phase).unwrap();
                kernel
                    .value_derivative(f32_to_f64_exact(x), &mut math)
                    .unwrap()
            };
            contains(value, &oracle.silu(q32(x)));
            contains(derivative, &oracle.silu_derivative(q32(x)));
            let envelope = kernel.enclose(x, x, &mut phase).unwrap();
            contains_envelope(&envelope, q32(x), &oracle.silu(q32(x)));
            if x > 0.0 {
                assert!(derivative.lo > 1.0);
                assert_eq!(envelope.slope().to_bits(), 1.0f32.to_bits());
            } else {
                assert!(derivative.hi < 0.0);
            }
        }
        assert_eq!(budget.receipt().unwrap().admitted, 1);
        assert_eq!(
            budget.receipt().unwrap().work_reserved,
            request(4).unwrap().work
        );
    }

    #[test]
    fn original_affine_activation_arbitrary_stored_slope_residual_is_paid() {
        let oracle = Oracle::new();
        for profile in [
            MathematicalActivation::GeluErf,
            MathematicalActivation::SiLu,
        ] {
            let (_, mut phase, kernel) = prepared(profile, 6);
            for slope in [-32.0, 0.0, 64.0] {
                let envelope = kernel
                    .enclose_with_slope(0.5, 1.5, Some(slope), &mut phase)
                    .unwrap();
                assert_eq!(envelope.slope().to_bits(), slope.to_bits());
                for x in [0.5, 1.0, 1.5] {
                    let x = q32(x);
                    contains_envelope(&envelope, x.clone(), &oracle.value(profile, x));
                }
                // Omitting r*|D-a| with a deliberately wrong stored slope
                // fails an actual endpoint. Thus the test requires the paid
                // slope residual, rather than just a loose image interval.
                let mut math = Math::new(&mut phase).unwrap();
                let (center, _) = kernel.value_derivative(1.0, &mut math).unwrap();
                let upper_without_residual =
                    q32(slope) * ratio(1, 2) + q64(center.hi) - q32(slope) + ratio(1, 8);
                let lower_without_residual =
                    q32(slope) * ratio(1, 2) + q64(center.lo) - q32(slope) - ratio(1, 8);
                let endpoint = oracle.value(profile, ratio(1, 2));
                if slope < -1.0 {
                    assert!(lower_without_residual > endpoint.hi);
                } else if slope > 2.0 {
                    assert!(upper_without_residual < endpoint.lo);
                }
            }
        }
    }

    #[test]
    fn original_affine_activation_subnormal_bits_zero_tail_and_overflow_hull() {
        for profile in [
            MathematicalActivation::GeluErf,
            MathematicalActivation::SiLu,
        ] {
            let (_, mut phase, kernel) = prepared(profile, 6);
            for zero in [0.0, -0.0] {
                let envelope = kernel.enclose(zero, zero, &mut phase).unwrap();
                assert_eq!(envelope.slope, 0.5);
                assert_eq!(envelope.lower_intercept, 0.0);
                assert_eq!(envelope.upper_intercept, 0.0);
                assert_eq!(envelope.image(), (0.0, 0.0));
                assert_eq!(envelope.interval().0.to_bits(), zero.to_bits());
            }
            let tiny = f32::from_bits(1);
            let envelope = kernel.enclose(-tiny, tiny, &mut phase).unwrap();
            assert_eq!(q32(envelope.lower), -q32(tiny));
            assert_eq!(q32(envelope.upper), q32(tiny));
            // Both exact profiles have f(0)=0, f'(0)=1/2 and M<=1.
            // The independent rational inequality below puts the Taylor
            // remainder strictly inside .01*|x| for |x|<=2^-149.
            assert!(q32(tiny) * q32(tiny) / integer(2) < q32(tiny) / integer(100));
            let positive = RationalInterval {
                lo: q32(tiny) * ratio(49, 100),
                hi: q32(tiny) * ratio(51, 100),
            };
            let negative = RationalInterval {
                lo: -positive.hi.clone(),
                hi: -positive.lo.clone(),
            };
            contains_envelope(&envelope, q32(tiny), &positive);
            contains_envelope(&envelope, -q32(tiny), &negative);
            let tail = kernel.enclose(-1024.0, -1024.0, &mut phase).unwrap();
            // Underflow handling never silently identifies a negative tail
            // with exact zero. A nonzero outward lower endpoint survives.
            assert!(tail.image_lower < 0.0);
            assert!(tail.image_upper <= 0.0);
            let huge = kernel.enclose(-f32::MAX, f32::MAX, &mut phase).unwrap();
            assert_eq!(huge.kind(), EnvelopeKind::GlobalRange);
            assert_eq!(huge.slope, 0.0);
            assert_eq!(huge.intercepts(), (-f32::MAX, f32::MAX));
            assert_eq!(huge.image(), (-f32::MAX, f32::MAX));
            // A subnormal proposal is deliberately stored as zero. Its full
            // loss is included in the derivative residual calculation.
            let flushed = kernel
                .enclose_with_slope(0.5, 1.0, Some(tiny), &mut phase)
                .unwrap();
            assert_eq!(flushed.slope().to_bits(), 0);
        }
    }

    #[test]
    fn original_affine_activation_poison_and_wrong_semantics_are_refused() {
        let (_, mut phase, mut kernel) = prepared(MathematicalActivation::GeluErf, 10);
        for (l, u) in [
            (f32::NAN, 1.0),
            (0.0, f32::INFINITY),
            (f32::NEG_INFINITY, 1.0),
            (2.0, 1.0),
        ] {
            assert!(kernel.enclose(l, u, &mut phase).is_err());
        }
        for slope in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(kernel
                .enclose_with_slope(0.0, 1.0, Some(slope), &mut phase)
                .is_err());
        }
        for profile in [
            "gelu-tanh-exact-real/v1",
            "gelu",
            "torch-gelu",
            "silu-binary32/v1",
            "",
        ] {
            assert!(MathematicalActivation::from_declared(profile, &mut phase).is_err());
        }
        assert_eq!(
            MathematicalActivation::from_declared("gelu-erf-exact-real/v1", &mut phase).unwrap(),
            MathematicalActivation::GeluErf
        );
        assert_eq!(
            MathematicalActivation::from_declared("silu-exact-real/v1", &mut phase).unwrap(),
            MathematicalActivation::SiLu
        );
        kernel.root_pi = I::new(f64::NAN, 2.0);
        assert!(kernel.enclose(0.0, 1.0, &mut phase).is_err());
    }

    #[test]
    fn original_affine_activation_original_phase_caps_no_refill_and_last_owner() {
        assert!(request(MAX_ENVELOPES + 1).is_err());
        assert!(request(usize::MAX).is_err());
        let mut req = request(1).unwrap();
        req.work = 128;
        let (budget, mut phase) = admitted(req);
        assert!(CheckedActivation::prepare(MathematicalActivation::GeluErf, &mut phase).is_err());
        assert_eq!(budget.receipt().unwrap().admitted, 1);
        assert_eq!(budget.receipt().unwrap().work_reserved, 128);
        drop(phase);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
        assert!(budget.lease(req).is_err());
        assert_eq!(budget.receipt().unwrap().work_reserved, 128);

        let mut req = request(0).unwrap();
        req.scratch_bytes -= 1;
        let (_, mut phase) = admitted(req);
        assert!(CheckedActivation::prepare(MathematicalActivation::SiLu, &mut phase).is_err());
        let mut req = request(0).unwrap();
        req.persistent_bytes = size_of::<CheckedActivation>() - 1;
        let (_, mut phase) = admitted(req);
        assert!(CheckedActivation::prepare(MathematicalActivation::SiLu, &mut phase).is_err());

        // Returned stack payloads are never granted a new resource authority.
        // A future private report owner must keep the original reservation.
        let req = request(1).unwrap();
        let (budget, mut phase, kernel) = prepared(MathematicalActivation::SiLu, 1);
        let envelope = kernel.enclose(-1.0, 1.0, &mut phase).unwrap();
        let reservation = phase.commit().unwrap();
        let last = reservation.clone();
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
        assert_eq!(
            budget.receipt().unwrap().live_persistent_bytes,
            req.persistent_bytes
        );
        drop(reservation);
        assert_eq!(
            budget.receipt().unwrap().live_persistent_bytes,
            req.persistent_bytes
        );
        let _ = envelope;
        drop(last);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
        assert_eq!(budget.receipt().unwrap().work_reserved, req.work);
    }

    #[test]
    fn original_affine_activation_foreign_phase_refuses_before_consuming_original_credits() {
        for same_ledger in [false, true] {
            let original_request = request(1).unwrap();
            let foreign_request = WorkRequest {
                scratch_bytes: SCRATCH,
                persistent_bytes: 1024,
                work: 16,
                proposals: 0,
            };
            let original_budget = WorkBudget::new(
                None,
                original_request.scratch_bytes
                    + if same_ledger {
                        foreign_request.scratch_bytes
                    } else {
                        0
                    },
                original_request.persistent_bytes
                    + if same_ledger {
                        foreign_request.persistent_bytes
                    } else {
                        0
                    },
                original_request.work + if same_ledger { foreign_request.work } else { 0 },
                0,
                Duration::ZERO,
            )
            .unwrap();
            let foreign_budget = if same_ledger {
                original_budget.clone()
            } else {
                WorkBudget::new(
                    None,
                    foreign_request.scratch_bytes,
                    foreign_request.persistent_bytes,
                    foreign_request.work,
                    0,
                    Duration::ZERO,
                )
                .unwrap()
            };
            let mut original_phase = OriginalAffinePhase::admit(
                &original_budget,
                LeaseClass::Optional,
                original_request,
                None,
            )
            .unwrap();
            let kernel =
                CheckedActivation::prepare(MathematicalActivation::GeluErf, &mut original_phase)
                    .unwrap();
            let mut foreign_phase = OriginalAffinePhase::admit(
                &foreign_budget,
                LeaseClass::Optional,
                foreign_request,
                None,
            )
            .unwrap();
            let before = foreign_budget.receipt().unwrap();
            let error = kernel.enclose(0.5, 1.5, &mut foreign_phase).unwrap_err();
            assert!(matches!(error, NyError::UnsupportedConfiguration(ref text)
                if text.contains("belongs to another phase")));
            // Private helper entry points also retain the exact preparation
            // identity. Creating Math performs no arithmetic or retention.
            {
                let mut math = Math::new(&mut foreign_phase).unwrap();
                assert!(kernel.validate(&mut math).is_err());
                assert!(kernel.value_derivative(0.5, &mut math).is_err());
                assert!(kernel.erfc(I::point(0.5), &mut math).is_err());
                assert!(kernel.erfc_point(0.5, &mut math).is_err());
            }
            assert_eq!(foreign_budget.receipt().unwrap(), before);
            // Receipts count reservations. Spend all private credits here to
            // additionally prove the foreign attempt consumed none of them.
            foreign_phase.charge_work(foreign_request.work).unwrap();
            foreign_phase
                .retain_metadata(foreign_request.persistent_bytes)
                .unwrap();
            assert_eq!(foreign_budget.receipt().unwrap(), before);
            drop(foreign_phase);
            assert!(kernel.enclose(0.5, 1.5, &mut original_phase).is_ok());
            drop(original_phase);
            assert_eq!(original_budget.receipt().unwrap().live_persistent_bytes, 0);
            assert_eq!(foreign_budget.receipt().unwrap().live_persistent_bytes, 0);
        }
    }

    #[test]
    fn original_affine_activation_same_phase_report_retains_prepared_claim_through_last_clone() {
        use crate::certify::work::PersistentReservation;
        use std::sync::Arc;

        struct HeldReport {
            _kernel: CheckedActivation,
            _envelope: ScalarEnvelope,
            _reservation: PersistentReservation,
        }
        let req = request(1).unwrap();
        let (budget, mut phase, kernel) = prepared(MathematicalActivation::SiLu, 1);
        let envelope = kernel.enclose(-1.0, 1.0, &mut phase).unwrap();
        // Both payloads were already retained by preparation/enclosure. Price
        // this test report's metadata/control allocation on the same phase.
        phase
            .retain_metadata(size_of::<PersistentReservation>() + 128)
            .unwrap();
        let reservation = phase.commit().unwrap();
        let report = Arc::new(HeldReport {
            _kernel: kernel,
            _envelope: envelope,
            _reservation: reservation,
        });
        let last = Arc::clone(&report);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
        assert_eq!(
            budget.receipt().unwrap().live_persistent_bytes,
            req.persistent_bytes
        );
        drop(report);
        assert_eq!(
            budget.receipt().unwrap().live_persistent_bytes,
            req.persistent_bytes
        );
        drop(last);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
        assert_eq!(budget.receipt().unwrap().work_reserved, req.work);
        assert_eq!(budget.receipt().unwrap().completed, 1);
    }

    #[test]
    fn original_affine_activation_cancellation_during_paid_recurrence_and_deadline() {
        let req = request(1).unwrap();
        let cancellation = CancellationToken::default();
        let memory = MemoryAuthority::new(req.scratch_bytes + req.persistent_bytes);
        let budget = WorkBudget::new_in(
            &memory,
            &cancellation,
            WorkLimits {
                deadline: None,
                scratch_bytes: req.scratch_bytes,
                persistent_bytes: req.persistent_bytes,
                max_work: req.work,
                max_proposals: req.proposals,
                certificate_reserve: Duration::ZERO,
                reserve: WorkReserve::default(),
            },
        )
        .unwrap();
        let mut phase =
            OriginalAffinePhase::admit(&budget, LeaseClass::Optional, req, None).unwrap();
        let kernel =
            CheckedActivation::prepare(MathematicalActivation::GeluErf, &mut phase).unwrap();
        struct ClearHook;
        impl Drop for ClearHook {
            fn drop(&mut self) {
                RECURRENCE_CANCEL.with(|hook| *hook.borrow_mut() = None);
            }
        }
        let _clear = ClearHook;
        RECURRENCE_CANCEL.with(|hook| *hook.borrow_mut() = Some((8, cancellation.clone())));
        assert!(kernel.enclose(0.5, 1.5, &mut phase).is_err());
        assert!(cancellation.is_cancelled());
        RECURRENCE_CANCEL.with(|hook| assert!(hook.borrow().is_none()));
        assert_eq!(budget.receipt().unwrap().admitted, 1);
        assert_eq!(budget.receipt().unwrap().completed, 0);
        assert_eq!(budget.receipt().unwrap().work_reserved, req.work);
        assert_eq!(
            budget.receipt().unwrap().live_persistent_bytes,
            req.persistent_bytes
        );
        drop(phase);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
        assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
        assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
        // Even after release, the original allowance is not reset or replaced.
        assert!(budget.lease(req).is_err());

        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        // A fresh *test* authority with an expired explicit inherited deadline
        // charges its admitted phase before refusal, and releases that phase.
        let test_budget = WorkBudget::new(
            None,
            req.scratch_bytes,
            req.persistent_bytes,
            req.work,
            0,
            Duration::ZERO,
        )
        .unwrap();
        assert!(
            OriginalAffinePhase::admit(&test_budget, LeaseClass::Replay, req, Some(expired))
                .is_err()
        );
        assert_eq!(test_budget.receipt().unwrap().admitted, 1);
        assert_eq!(test_budget.receipt().unwrap().work_reserved, req.work);
        assert_eq!(test_budget.receipt().unwrap().live_persistent_bytes, 0);
        assert_eq!(budget.receipt().unwrap().work_reserved, req.work);
    }

    #[test]
    fn activation_construction_metadata_accepts_checked_taylor_and_global_hulls() {
        for profile in [
            MathematicalActivation::GeluErf,
            MathematicalActivation::SiLu,
        ] {
            let (_, mut phase, kernel) = prepared(profile, 4);
            let tiny = f32::from_bits(1);
            for (lower, upper) in [(0.0, 0.0), (-tiny, tiny), (-0.5, 0.5), (-1.0e30, 1.0e30)] {
                let envelope = kernel.enclose(lower, upper, &mut phase).unwrap();
                envelope.validate_construction_metadata(&mut phase).unwrap();
            }
        }
    }

    #[test]
    fn activation_construction_metadata_refuses_poisoned_centers_and_global_hulls() {
        let (_, mut phase, kernel) = prepared(MathematicalActivation::SiLu, 2);
        let mut taylor = kernel.enclose(-0.5, 0.5, &mut phase).unwrap();
        let center = taylor.center;
        for poison in [f32::NAN, f32::INFINITY, -1.0, 1.0] {
            taylor.center = poison;
            assert!(taylor.validate_construction_metadata(&mut phase).is_err());
        }
        taylor.center = center;
        taylor.validate_construction_metadata(&mut phase).unwrap();
        taylor.kind = EnvelopeKind::GlobalRange;
        assert!(taylor.validate_construction_metadata(&mut phase).is_err());

        let mut global = kernel.enclose(-1.0e30, 1.0e30, &mut phase).unwrap();
        assert_eq!(global.kind(), EnvelopeKind::GlobalRange);
        global.validate_construction_metadata(&mut phase).unwrap();
        global.slope = 1.0;
        assert!(global.validate_construction_metadata(&mut phase).is_err());
        global.slope = 0.0;
        let lower = global.lower_intercept;
        global.lower_intercept = 0.0;
        assert!(global.validate_construction_metadata(&mut phase).is_err());
        global.lower_intercept = lower;
        let upper = global.upper_intercept;
        global.upper_intercept = 0.0;
        assert!(global.validate_construction_metadata(&mut phase).is_err());
        global.upper_intercept = upper;
        let lower = global.image_lower;
        global.image_lower = 0.0;
        assert!(global.validate_construction_metadata(&mut phase).is_err());
        global.image_lower = lower;
        let upper = global.image_upper;
        global.image_upper = 0.0;
        assert!(global.validate_construction_metadata(&mut phase).is_err());
        global.image_upper = upper;
        global.validate_construction_metadata(&mut phase).unwrap();
    }
}
