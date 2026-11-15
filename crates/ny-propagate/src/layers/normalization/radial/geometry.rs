// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Checked normalization geometry and elementary square-root authority.

use super::{checkpoint, refusal, require_environment, I};
use crate::certify::work::WorkBudget;
use crate::layers::softmax::weighted_support::{down, up};
use ny_core::Result;

/// libm proposes endpoints; outward squares independently check both.
pub(super) fn sqrt_point(x: f64) -> Option<I> {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    // Floating comparisons can themselves erase a subnormal operand in DAZ
    // mode. Decide the original finite/sign/zero/subnormal profile by bits.
    if magnitude >= 0x7ff0_0000_0000_0000 || (bits >> 63 != 0 && magnitude != 0) {
        return None;
    }
    if magnitude == 0 {
        return Some(I::point(0.0));
    }
    if magnitude < 0x0010_0000_0000_0000 {
        // A rounded subnormal square cannot authorize the usual one-ulp
        // endpoint checks: many adjacent root proposals have the same square.
        // Scaling by 2^54 is exact and puts every positive subnormal into the
        // normal range. The recursive call therefore takes the normal branch.
        // Its root endpoints and their 2^-27 rescalings are all normal, so the
        // final multiplications are exact too. Squared containment transfers
        // directly from the independently checked scaled interval.
        let scaled_input = x * 18_014_398_509_481_984.0;
        // Bind the actual multiplication to the exact expected NORMAL value
        // by integer representation. DAZ->zero, any subnormal or altered scale
        // refuses BEFORE a recursive zero root could authorize an enclosure.
        if !scaled_subnormal_matches(magnitude, scaled_input) {
            return None;
        }
        let scaled = sqrt_point(scaled_input)?;
        let root_scale = f64::from_bits(0x3e40_0000_0000_0000); // exactly 2^-27
        return Some(I::new(scaled.lo * root_scale, scaled.hi * root_scale));
    }
    let proposal = x.sqrt();
    if !proposal.is_finite() {
        return None;
    }
    let (mut lower, mut upper) = (down(proposal).max(0.0), up(proposal));
    for _ in 0..8 {
        if up(lower * lower) <= x && down(upper * upper) >= x {
            return Some(I::new(lower, upper));
        }
        lower = down(lower).max(0.0);
        upper = up(upper);
    }
    None
}

/// Exact positive subnormal x*2^54, built without reading x as an FP operand.
fn scaled_subnormal_bits(magnitude: u64) -> Option<u64> {
    if magnitude == 0 || magnitude >= 0x0010_0000_0000_0000 {
        return None;
    }
    let leading = magnitude.ilog2();
    let exponent = u64::from(leading + 3) << 52;
    let fraction = (magnitude - (1_u64 << leading)) << (52 - leading);
    Some(exponent | fraction)
}

fn scaled_subnormal_matches(magnitude: u64, scaled: f64) -> bool {
    scaled_subnormal_bits(magnitude).is_some_and(|expected| scaled.to_bits() == expected)
}

pub(super) fn sqrt_interval(x: I) -> Option<I> {
    Some(I::new(sqrt_point(x.lo.max(0.0))?.lo, sqrt_point(x.hi)?.hi))
}

pub(super) fn square(x: I) -> I {
    if x.lo == 0.0 && x.hi == 0.0 {
        return I::point(0.0);
    }
    let minimum = if x.lo <= 0.0 && x.hi >= 0.0 {
        0.0
    } else {
        let endpoint = x.lo.abs().min(x.hi.abs());
        down(endpoint * endpoint).max(0.0)
    };
    I::new(minimum, up(x.abs_max() * x.abs_max()))
}

pub(super) fn norm(v: &[I], budget: Option<&WorkBudget>) -> Result<I> {
    checkpoint(budget)?;
    require_environment()?;
    let mut sum = I::point(0.0);
    for (i, &x) in v.iter().enumerate() {
        if i % 256 == 0 {
            checkpoint(budget)?;
        }
        sum = sum.add(square(x));
    }
    sqrt_interval(sum).ok_or_else(|| refusal("norm square-root check refused"))
}

pub(super) fn project(v: &[I], centered: bool, budget: Option<&WorkBudget>) -> Result<Vec<I>> {
    checkpoint(budget)?;
    if !centered {
        return Ok(v.to_vec());
    }
    let mut uniform = true;
    let mut sum = I::point(0.0);
    for (i, &x) in v.iter().enumerate() {
        if i % 256 == 0 {
            checkpoint(budget)?;
        }
        uniform &= x.lo == x.hi && x.lo == v[0].lo;
        sum = sum.add(x);
    }
    if uniform {
        return Ok(vec![I::point(0.0); v.len()]);
    }
    let mean = sum.div_positive(I::point(v.len() as f64));
    let mut result = Vec::with_capacity(v.len());
    for (i, &x) in v.iter().enumerate() {
        if i % 256 == 0 {
            checkpoint(budget)?;
        }
        result.push(x.sub(mean));
    }
    Ok(result)
}

pub(super) struct Geometry {
    pub(super) center: Vec<f64>,
    pub(super) radius: Vec<f64>,
    pub(super) z0: Vec<I>,
    pub(super) z_box: Vec<I>,
    pub(super) denominator: I,
    pub(super) t_min: f64,
    pub(super) rho2: f64,
    pub(super) root_n: I,
}

impl Geometry {
    pub(super) fn new(
        lower: &[f32],
        upper: &[f32],
        eps: f32,
        centered: bool,
        budget: Option<&WorkBudget>,
    ) -> Result<Self> {
        checkpoint(budget)?;
        require_environment()?;
        #[cfg(test)]
        CONSTRUCTIONS.with(|count| count.set(count.get() + 1));
        let n = lower.len();
        let mut center = Vec::with_capacity(n);
        let mut radius = Vec::with_capacity(n);
        let mut rho2 = I::point(0.0);
        for (i, (&l, &u)) in lower.iter().zip(upper).enumerate() {
            if i % 256 == 0 {
                checkpoint(budget)?;
            }
            if !l.is_finite() || !u.is_finite() || l > u {
                return Err(refusal("finite ordered input box required"));
            }
            let x0 = (l as f64) * 0.5 + (u as f64) * 0.5;
            let r = I::point(l as f64)
                .sub(I::point(x0))
                .abs_max()
                .max(I::point(u as f64).sub(I::point(x0)).abs_max());
            center.push(x0);
            radius.push(r);
            rho2 = rho2.add(I::point(r).mul(I::point(r)));
        }
        let z0 = project(
            &center.iter().copied().map(I::point).collect::<Vec<_>>(),
            centered,
            budget,
        )?;
        let input_box: Vec<I> = lower
            .iter()
            .zip(upper)
            .map(|(&l, &u)| I::new(l as f64, u as f64))
            .collect();
        let z_box = project(&input_box, centered, budget)?;
        let norm0 = norm(&z0, budget)?;
        let nf = I::point(n as f64);
        let mut norm02 = I::point(0.0);
        for (i, &x) in z0.iter().enumerate() {
            if i % 256 == 0 {
                checkpoint(budget)?;
            }
            norm02 = norm02.add(square(x));
        }
        let denominator = sqrt_interval(norm02.div_positive(nf).add(I::point(eps as f64)))
            .ok_or_else(|| refusal("center denominator could not be checked"))?;
        let rho = sqrt_point(rho2.hi).ok_or_else(|| refusal("box radius could not be checked"))?;
        // P is an orthogonal projection, hence ||P delta|| <= ||delta||.
        let distance = I::point(norm0.lo).sub(I::point(rho.hi)).lo.max(0.0);
        let mut denominator_lower2 = I::point(distance)
            .mul(I::point(distance))
            .div_positive(nf)
            .add(I::point(eps as f64))
            .lo;
        if !centered {
            let exact_box_norm_lower2 = input_box
                .iter()
                .fold(I::point(0.0), |s, &x| s.add(square(x)));
            denominator_lower2 = denominator_lower2.max(
                exact_box_norm_lower2
                    .div_positive(nf)
                    .add(I::point(eps as f64))
                    .lo,
            );
        }
        let t_min = sqrt_point(denominator_lower2.max(eps as f64))
            .ok_or_else(|| refusal("minimum denominator could not be checked"))?
            .lo;
        let root_n = sqrt_point(n as f64).ok_or_else(|| refusal("dimension root refused"))?;
        if t_min <= 0.0 || !denominator.finite() {
            return Err(refusal("positive denominator required"));
        }
        Ok(Self {
            center,
            radius,
            z0,
            z_box,
            denominator,
            t_min,
            rho2: rho2.hi,
            root_n,
        })
    }

    pub(super) fn output_abs(&self, gamma: f32, beta: f32, i: usize) -> f64 {
        let normalized_abs = I::point(self.z_box[i].abs_max())
            .div_positive(I::point(self.t_min))
            .hi
            .min(self.root_n.hi);
        I::point((gamma as f64).abs())
            .mul(I::point(normalized_abs))
            .add(I::point((beta as f64).abs()))
            .hi
    }
}

#[cfg(test)]
thread_local! {
    static CONSTRUCTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
pub(super) fn reset_constructions() {
    CONSTRUCTIONS.with(|count| count.set(0));
}
#[cfg(test)]
pub(super) fn constructions() -> usize {
    CONSTRUCTIONS.with(|count| count.get())
}

#[cfg(test)]
mod sqrt_tests {
    use super::*;
    use num_rational::BigRational;

    fn rational_squared_containment(x: f64) {
        let interval = sqrt_point(x).expect("finite nonnegative checked root");
        assert!(interval.finite() && interval.lo >= 0.0);
        let exact = BigRational::from_float(x).unwrap();
        let lower = BigRational::from_float(interval.lo).unwrap();
        let upper = BigRational::from_float(interval.hi).unwrap();
        assert!(&lower * &lower <= exact, "lower endpoint at {x:e}");
        assert!(&upper * &upper >= exact, "upper endpoint at {x:e}");
    }

    #[test]
    fn decoder_n12_checked_sqrt_subnormal_scale_binds_original_bits_and_refuses_flush() {
        // No FP mode mutation: exercise the actual guard with a flushed/wrong
        // multiplication result and independently bind the valid result with
        // exact rational original arithmetic, including all mantissa scales.
        let scale = BigRational::from_integer((1_u64 << 54).into());
        for magnitude in [
            1,
            3,
            17,
            (1 << 20) + 1,
            (1 << 44) + 1,
            0x000f_ffff_ffff_ffff,
        ] {
            let expected = f64::from_bits(scaled_subnormal_bits(magnitude).unwrap());
            assert_eq!(
                BigRational::from_float(expected).unwrap(),
                BigRational::from_float(f64::from_bits(magnitude)).unwrap() * &scale
            );
            assert!(scaled_subnormal_matches(magnitude, expected));
            assert!(expected.to_bits() >= f64::MIN_POSITIVE.to_bits());
            for wrong in [
                0.0,
                -0.0,
                f64::from_bits(1),
                -expected,
                f64::from_bits(expected.to_bits() + 1),
                f64::INFINITY,
                f64::NAN,
            ] {
                assert!(!scaled_subnormal_matches(magnitude, wrong));
            }
        }
        assert!(scaled_subnormal_bits(0).is_none());
        assert!(scaled_subnormal_bits(f64::MIN_POSITIVE.to_bits()).is_none());
        assert!(scaled_subnormal_bits(0x8000_0000_0000_0001).is_none());
    }

    #[test]
    fn decoder_n12_checked_sqrt_subnormal_endpoints_have_rational_squared_authority() {
        ny_core::require_f64_interval_proof_environment().unwrap();
        for bits in [
            1,
            2,
            3,
            5,
            17,
            1_023,
            (1 << 20) + 1,
            (1 << 44) + 1,
            0x0008_0000_0000_0001,
            0x000f_ffff_ffff_fffd,
            0x000f_ffff_ffff_ffff,
        ] {
            rational_squared_containment(f64::from_bits(bits));
        }
    }

    #[test]
    fn decoder_n12_checked_sqrt_normal_boundary_and_ordinary_controls() {
        ny_core::require_f64_interval_proof_environment().unwrap();
        for x in [
            f64::MIN_POSITIVE,
            f64::from_bits(f64::MIN_POSITIVE.to_bits() + 1),
            0.25,
            1.0,
            2.0,
            1e-5,
            f64::MAX,
        ] {
            rational_squared_containment(x);
        }
    }

    #[test]
    fn decoder_n12_checked_sqrt_zero_and_invalid_profile_is_unchanged() {
        for zero in [0.0, -0.0] {
            let interval = sqrt_point(zero).unwrap();
            assert_eq!(interval.lo.to_bits(), 0.0_f64.to_bits());
            assert_eq!(interval.hi.to_bits(), 0.0_f64.to_bits());
        }
        for invalid in [
            -f64::from_bits(1),
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            assert!(sqrt_point(invalid).is_none());
        }
    }
}
