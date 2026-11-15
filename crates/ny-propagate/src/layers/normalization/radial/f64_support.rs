// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original binary64 directional support without binary32 carriers.
//!
//! This checks the mathematical real normalization only. Callers admit its
//! bounded buffers and charge/poll every coordinate scan on their existing
//! lease. Original binary64 differences, products, projection, beta and bias
//! remain intervals throughout. The existing checked square-root primitive
//! authorizes every root; platform sqrt alone supplies no authority.

use super::geometry::{sqrt_interval, sqrt_point, square};
use super::I;

#[derive(Debug)]
pub(crate) enum SupportError<E> {
    Arithmetic(&'static str),
    Poll(E),
}
type Result<T, E> = std::result::Result<T, SupportError<E>>;

/// Decision demand, rather than a request to minimize a public support value.
/// Full preserves the original independently checked candidate minimum.
#[derive(Clone, Copy)]
pub(crate) enum SupportDemand {
    Full,
    StrictNegative,
    AtMost(f64),
}
impl SupportDemand {
    fn met(self, upper: f64) -> bool {
        match self {
            Self::Full => false,
            Self::StrictNegative => upper < 0.0,
            Self::AtMost(threshold) => upper <= threshold,
        }
    }
}

fn poll<E>(
    check: &mut impl FnMut(usize) -> std::result::Result<(), E>,
    count: usize,
) -> Result<(), E> {
    check(count).map_err(SupportError::Poll)
}
fn storage<T, E>(count: usize) -> Result<Vec<T>, E> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| SupportError::Arithmetic("support allocation"))?;
    Ok(values)
}
fn finite<E>(value: I, why: &'static str) -> Result<I, E> {
    if value.finite() {
        Ok(value)
    } else {
        Err(SupportError::Arithmetic(why))
    }
}
fn norm<E>(v: &[I], check: &mut impl FnMut(usize) -> std::result::Result<(), E>) -> Result<I, E> {
    let mut sum = I::point(0.0);
    for block in v.chunks(256) {
        poll(check, block.len())?;
        for &value in block {
            sum = sum.add(square(value));
        }
    }
    sqrt_interval(sum).ok_or(SupportError::Arithmetic("support norm"))
}
fn project<E>(
    v: &[I],
    centered: bool,
    check: &mut impl FnMut(usize) -> std::result::Result<(), E>,
) -> Result<Vec<I>, E> {
    let mut result = storage(v.len())?;
    if !centered {
        for block in v.chunks(256) {
            poll(check, block.len())?;
            result.extend_from_slice(block);
        }
        return Ok(result);
    }
    let mut sum = I::point(0.0);
    let mut uniform = true;
    for block in v.chunks(256) {
        poll(check, block.len())?;
        for &value in block {
            uniform &= value.lo == value.hi && value.lo == v[0].lo;
            sum = sum.add(value);
        }
    }
    // Equality of rounded proposals is insufficient. Only exact singleton
    // intervals with identical values authorize this algebraic cancellation.
    let mean = if uniform {
        I::point(v[0].lo)
    } else {
        finite(
            sum.div_positive(I::point(v.len() as f64)),
            "support projection mean",
        )?
    };
    for block in v.chunks(256) {
        poll(check, block.len())?;
        for &value in block {
            result.push(finite(value.sub(mean), "support projection")?);
        }
    }
    Ok(result)
}

/// Region geometry shared by all original competitors in one admitted scan.
pub(crate) struct GeometryF64 {
    center: Vec<f64>,
    radius: Vec<f64>,
    delta: Vec<I>,
    z0: Vec<I>,
    z_box: Vec<I>,
    denominator: I,
    denominator_box: I,
    t_min: f64,
    rho2: f64,
    root_n: I,
    centered: bool,
}
impl GeometryF64 {
    /// Dimension of this checked original pre-normalization region.
    pub(crate) fn hidden_size(&self) -> usize {
        self.center.len()
    }

    pub(crate) fn new<E>(
        lower: &[f64],
        upper: &[f64],
        epsilon: f64,
        centered: bool,
        check: &mut impl FnMut(usize) -> std::result::Result<(), E>,
    ) -> Result<Self, E> {
        poll(check, 0)?;
        ny_core::require_f64_interval_proof_environment()
            .map_err(|_| SupportError::Arithmetic("support binary64 proof environment"))?;
        let n = lower.len();
        if n == 0
            || n != upper.len()
            || n > (1 << 24)
            || !epsilon.is_finite()
            || epsilon < super::super::NORMALIZATION_MIN_EPS as f64
        {
            return Err(SupportError::Arithmetic("support shape/epsilon domain"));
        }
        let mut center = storage(n)?;
        let mut radius = storage(n)?;
        let mut delta = storage(n)?;
        let mut input_box = storage(n)?;
        let mut rho2 = I::point(0.0);
        for start in (0..n).step_by(256) {
            let end = (start + 256).min(n);
            poll(check, end - start)?;
            for i in start..end {
                let (l, u) = (lower[i], upper[i]);
                if !l.is_finite() || !u.is_finite() || l > u {
                    return Err(SupportError::Arithmetic("support finite ordered box"));
                }
                // Halving a binary64 subnormal can round outside a singleton
                // region. The Taylor tube must remain inside the supplied box.
                let anchor = (l * 0.5 + u * 0.5).clamp(l, u);
                if !anchor.is_finite() {
                    return Err(SupportError::Arithmetic("support anchor"));
                }
                let d = finite(
                    I::new(l, u).sub(I::point(anchor)),
                    "support asymmetric displacement",
                )?;
                let r = d.abs_max();
                center.push(anchor);
                radius.push(r);
                delta.push(d);
                input_box.push(I::new(l, u));
                rho2 = rho2.add(square(I::point(r)));
            }
        }
        let mut center_i = storage(n)?;
        for block in center.chunks(256) {
            poll(check, block.len())?;
            center_i.extend(block.iter().copied().map(I::point));
        }
        let z0 = project(&center_i, centered, check)?;
        drop(center_i);
        let z_box = project(&input_box, centered, check)?;
        let norm0 = norm(&z0, check)?;
        let nf = I::point(n as f64);
        let mut norm02 = I::point(0.0);
        let mut box_norm2 = I::point(0.0);
        for start in (0..n).step_by(256) {
            let end = (start + 256).min(n);
            poll(check, end - start)?;
            for i in start..end {
                norm02 = norm02.add(square(z0[i]));
                box_norm2 = box_norm2.add(square(z_box[i]));
            }
        }
        let denominator = sqrt_interval(norm02.div_positive(nf).add(I::point(epsilon)))
            .ok_or(SupportError::Arithmetic("support anchor denominator"))?;
        let denominator_box = sqrt_interval(box_norm2.div_positive(nf).add(I::point(epsilon)))
            .ok_or(SupportError::Arithmetic("support box denominator"))?;
        let rho = sqrt_point(rho2.hi).ok_or(SupportError::Arithmetic("support radius"))?;
        // P is orthogonal; ||P delta|| <= ||delta|| over the complete tube.
        let distance = I::point(norm0.lo).sub(I::point(rho.hi)).lo.max(0.0);
        let lower2 = square(I::point(distance))
            .div_positive(nf)
            .add(I::point(epsilon))
            .lo
            .max(box_norm2.div_positive(nf).add(I::point(epsilon)).lo)
            .max(epsilon);
        let t_min = sqrt_point(lower2)
            .ok_or(SupportError::Arithmetic("support denominator lower"))?
            .lo;
        let root_n =
            sqrt_point(n as f64).ok_or(SupportError::Arithmetic("support dimension root"))?;
        if t_min <= 0.0 || !denominator.finite() || !denominator_box.finite() {
            return Err(SupportError::Arithmetic("support positive denominator"));
        }
        Ok(Self {
            center,
            radius,
            delta,
            z0,
            z_box,
            denominator,
            denominator_box: I::new(t_min, denominator_box.hi),
            t_min,
            rho2: rho2.hi,
            root_n,
            centered,
        })
    }

    /// Checked learned normalization at the clamped expansion anchor. This is
    /// an enclosure of one original point, not the center of an IBP output.
    /// The caller prices the returned interval vector on its existing lease.
    pub(crate) fn normalized_anchor_image<E>(
        &self,
        gamma: &[f64],
        beta: Option<&[f64]>,
        check: &mut impl FnMut(usize) -> std::result::Result<(), E>,
    ) -> Result<Vec<I>, E> {
        poll(check, 0)?;
        let n = self.center.len();
        if gamma.len() != n || beta.is_some_and(|beta| beta.len() != n) {
            return Err(SupportError::Arithmetic("support anchor parameter shape"));
        }
        let mut image = storage(n)?;
        if image
            .capacity()
            .checked_mul(size_of::<I>())
            .is_none_or(|bytes| {
                n.checked_mul(size_of::<I>())
                    .and_then(|bytes| bytes.checked_add(128))
                    .is_none_or(|allowance| bytes > allowance)
            })
        {
            return Err(SupportError::Arithmetic("support anchor vector capacity"));
        }
        for start in (0..n).step_by(256) {
            let end = (start + 256).min(n);
            poll(check, end - start)?;
            for i in start..end {
                if !gamma[i].is_finite() || beta.is_some_and(|beta| !beta[i].is_finite()) {
                    return Err(SupportError::Arithmetic("support finite anchor parameters"));
                }
                image.push(finite(
                    self.z0[i]
                        .div_positive(self.denominator)
                        .mul(I::point(gamma[i]))
                        .add(I::point(beta.map_or(0.0, |beta| beta[i]))),
                    "support learned anchor image",
                )?);
            }
        }
        poll(check, 0)?;
        Ok(image)
    }

    /// Outward norm of P diag(gamma)(row-target). Recomputing the original
    /// differences in the second scan avoids a derived coefficient buffer.
    pub(crate) fn projected_direction_norm<E>(
        &self,
        row: &[f64],
        target: &[f64],
        gamma: &[f64],
        check: &mut impl FnMut(usize) -> std::result::Result<(), E>,
    ) -> Result<I, E> {
        poll(check, 0)?;
        let n = self.center.len();
        if row.len() != n || target.len() != n || gamma.len() != n {
            return Err(SupportError::Arithmetic("support group direction shape"));
        }
        let mut sum = I::point(0.0);
        let mut first = None;
        let mut uniform = true;
        for start in (0..n).step_by(256) {
            let end = (start + 256).min(n);
            poll(check, end - start)?;
            for i in start..end {
                if !row[i].is_finite() || !target[i].is_finite() || !gamma[i].is_finite() {
                    return Err(SupportError::Arithmetic("support finite group direction"));
                }
                let value = finite(
                    I::point(row[i])
                        .sub(I::point(target[i]))
                        .mul(I::point(gamma[i])),
                    "support group direction product",
                )?;
                let reference = *first.get_or_insert(value.lo);
                uniform &= value.lo == value.hi && value.lo == reference;
                sum = finite(sum.add(value), "support group direction sum")?;
            }
        }
        let mean = if !self.centered {
            I::point(0.0)
        } else if uniform {
            I::point(first.ok_or(SupportError::Arithmetic("support empty group direction"))?)
        } else {
            finite(
                sum.div_positive(I::point(n as f64)),
                "support group direction mean",
            )?
        };
        let mut norm2 = I::point(0.0);
        for start in (0..n).step_by(256) {
            let end = (start + 256).min(n);
            poll(check, end - start)?;
            for i in start..end {
                let value = finite(
                    I::point(row[i])
                        .sub(I::point(target[i]))
                        .mul(I::point(gamma[i]))
                        .sub(mean),
                    "support projected group direction",
                )?;
                norm2 = finite(norm2.add(square(value)), "support group direction norm2")?;
            }
        }
        let result =
            sqrt_interval(norm2).ok_or(SupportError::Arithmetic("support group direction norm"))?;
        poll(check, 0)?;
        Ok(result)
    }

    /// Whole-region displacement of the unlearned normalized vector. For
    /// f(x)=Px/sqrt(||Px||²/n+eps), ||Df|| <= 1/t on every segment in the box:
    /// transverse eigenvalues are 1/t and the radial eigenvalue is eps/t³.
    /// Also ||f|| <= sqrt(n), including strictly positive epsilon.
    pub(crate) fn normalized_displacement_upper<E>(
        &self,
        check: &mut impl FnMut(usize) -> std::result::Result<(), E>,
    ) -> Result<f64, E> {
        poll(check, 0)?;
        if self.centered && self.center.len() == 1 {
            return Ok(0.0);
        }
        let rho = sqrt_point(self.rho2).ok_or(SupportError::Arithmetic(
            "support group displacement radius",
        ))?;
        let local = finite(
            rho.div_positive(I::point(self.t_min)),
            "support group displacement tube",
        )?;
        let global = finite(
            self.root_n.mul(I::point(2.0)),
            "support group displacement global",
        )?;
        let upper = local.hi.min(global.hi);
        if !upper.is_finite() || upper < 0.0 {
            return Err(SupportError::Arithmetic(
                "support finite group displacement",
            ));
        }
        poll(check, 0)?;
        Ok(upper)
    }

    pub(crate) fn upper<E>(
        &self,
        row: &[f64],
        target: &[f64],
        gamma: &[f64],
        beta: Option<&[f64]>,
        row_bias: f64,
        target_bias: f64,
        check: &mut impl FnMut(usize) -> std::result::Result<(), E>,
    ) -> Result<SupportUpper, E> {
        self.upper_for(
            row,
            target,
            gamma,
            beta,
            row_bias,
            target_bias,
            SupportDemand::Full,
            check,
        )
    }

    /// A conditional head query can stop at any independently checked support
    /// meeting its strict decision or replay threshold. It still transforms
    /// every original coordinate. The generic full-support API remains eager.
    pub(crate) fn upper_for<E>(
        &self,
        row: &[f64],
        target: &[f64],
        gamma: &[f64],
        beta: Option<&[f64]>,
        row_bias: f64,
        target_bias: f64,
        demand: SupportDemand,
        check: &mut impl FnMut(usize) -> std::result::Result<(), E>,
    ) -> Result<SupportUpper, E> {
        poll(check, 0)?;
        if matches!(demand, SupportDemand::AtMost(value) if !value.is_finite()) {
            return Err(SupportError::Arithmetic("support finite replay threshold"));
        }
        let n = self.center.len();
        if row.len() != n
            || target.len() != n
            || gamma.len() != n
            || beta.is_some_and(|b| b.len() != n)
        {
            return Err(SupportError::Arithmetic("support direction shape"));
        }
        if !row_bias.is_finite() || !target_bias.is_finite() {
            return Err(SupportError::Arithmetic("support finite original bias"));
        }
        let mut transformed = storage(n)?;
        let mut constant = finite(
            I::point(row_bias).sub(I::point(target_bias)),
            "support bias difference",
        )?;
        for start in (0..n).step_by(256) {
            let end = (start + 256).min(n);
            poll(check, end - start)?;
            for i in start..end {
                // Validate originals before algebraic zero/identity shortcuts:
                // Inf-Inf and zero*Inf may otherwise hide invalid premises.
                if !row[i].is_finite()
                    || !target[i].is_finite()
                    || !gamma[i].is_finite()
                    || beta.is_some_and(|b| !b[i].is_finite())
                {
                    return Err(SupportError::Arithmetic(
                        "support finite original direction/parameters",
                    ));
                }
                let difference = finite(
                    I::point(row[i]).sub(I::point(target[i])),
                    "support original difference",
                )?;
                transformed.push(finite(
                    difference.mul(I::point(gamma[i])),
                    "support learned-scale product",
                )?);
                if let Some(beta) = beta {
                    constant = finite(
                        constant.add(difference.mul(I::point(beta[i]))),
                        "support learned shift constant",
                    )?;
                }
            }
        }
        let a = project(&transformed, self.centered, check)?;
        drop(transformed);
        let mut projection_zero = true;
        let mut dense_dot = I::point(0.0);
        for start in (0..n).step_by(256) {
            let end = (start + 256).min(n);
            poll(check, end - start)?;
            for i in start..end {
                projection_zero &= a[i].lo == 0.0 && a[i].hi == 0.0;
                dense_dot = dense_dot.add(a[i].mul(self.z_box[i]));
            }
        }
        let mut dense_upper = finite(
            dense_dot.div_positive(self.denominator_box).add(constant),
            "support dense fallback",
        )?
        .hi;
        if demand.met(dense_upper) {
            poll(check, 0)?;
            return Ok(SupportUpper {
                upper: dense_upper,
                dense_upper,
                taylor_upper: None,
                projection_zero,
                dense_screened: true,
                cauchy_screened: false,
                taylor_attempted: false,
            });
        }
        // This is an independent checked Cauchy bound. LayerNorm removes the
        // mean direction; no division by gamma, including zero gamma entries.
        let a_norm = match norm(&a, check) {
            Ok(value) => Some(value),
            Err(SupportError::Arithmetic(_)) => None,
            Err(error) => return Err(error),
        };
        if let Some(a_norm) = a_norm {
            let global = a_norm.mul(self.root_n).add(constant);
            if global.finite() {
                dense_upper = dense_upper.min(global.hi);
            }
        }
        if demand.met(dense_upper) {
            poll(check, 0)?;
            return Ok(SupportUpper {
                upper: dense_upper,
                dense_upper,
                taylor_upper: None,
                projection_zero,
                dense_screened: false,
                cauchy_screened: true,
                taylor_attempted: false,
            });
        }
        let taylor_attempted = a_norm.is_some();
        let taylor_upper = if let Some(a_norm) = a_norm {
            match self.taylor(&a, a_norm, constant, check) {
                Ok(value) => Some(value),
                Err(SupportError::Arithmetic(_)) => None,
                Err(error) => return Err(error),
            }
        } else {
            None
        };
        let upper = taylor_upper.map_or(dense_upper, |value| value.min(dense_upper));
        if !upper.is_finite() {
            return Err(SupportError::Arithmetic("support finite upper"));
        }
        poll(check, 0)?;
        Ok(SupportUpper {
            upper,
            dense_upper,
            taylor_upper,
            projection_zero,
            dense_screened: false,
            cauchy_screened: false,
            taylor_attempted,
        })
    }

    fn taylor<E>(
        &self,
        a: &[I],
        a_norm: I,
        constant: I,
        check: &mut impl FnMut(usize) -> std::result::Result<(), E>,
    ) -> Result<f64, E> {
        let n = a.len();
        let nf = I::point(n as f64);
        let mut dot = I::point(0.0);
        for start in (0..n).step_by(256) {
            let end = (start + 256).min(n);
            poll(check, end - start)?;
            for i in start..end {
                dot = dot.add(a[i].mul(self.z0[i]));
            }
        }
        let center_value = dot.div_positive(self.denominator).add(constant);
        let t3 = self.denominator.mul(self.denominator).mul(self.denominator);
        let factor = dot.div_positive(nf.mul(t3));
        let mut linear = I::point(0.0);
        for start in (0..n).step_by(256) {
            let end = (start + 256).min(n);
            poll(check, end - start)?;
            for i in start..end {
                let gradient = a[i]
                    .div_positive(self.denominator)
                    .sub(factor.mul(self.z0[i]));
                linear = linear.add(gradient.mul(self.delta[i]));
            }
        }
        let tmin = I::point(self.t_min);
        let tmin2 = tmin.mul(tmin);
        let global = I::point(3.0)
            .mul(I::point(a_norm.hi))
            .mul(I::point(self.rho2))
            .div_positive(I::point(self.root_n.lo).mul(tmin2));
        let mut remainder = if global.finite() {
            global.hi
        } else {
            f64::INFINITY
        };
        let mut a_v = I::point(0.0);
        let mut z0_v = I::point(0.0);
        let mut a_z_box = I::point(0.0);
        for start in (0..n).step_by(256) {
            let end = (start + 256).min(n);
            poll(check, end - start)?;
            for i in start..end {
                a_v = a_v.add(I::point(a[i].abs_max()).mul(I::point(self.radius[i])));
                z0_v = z0_v.add(I::point(self.z0[i].abs_max()).mul(I::point(self.radius[i])));
                a_z_box = a_z_box.add(a[i].mul(self.z_box[i]));
            }
        }
        let z_v = z0_v.add(I::point(self.rho2));
        let max_a_z = I::point(a_z_box.abs_max());
        let tmin3 = tmin2.mul(tmin);
        let tmin5 = tmin3.mul(tmin2);
        let directional = a_v
            .mul(z_v)
            .div_positive(nf.mul(tmin3))
            .add(
                I::point(0.5)
                    .mul(max_a_z)
                    .mul(I::point(self.rho2))
                    .div_positive(nf.mul(tmin3)),
            )
            .add(
                I::point(1.5)
                    .mul(max_a_z)
                    .mul(z_v)
                    .mul(z_v)
                    .div_positive(nf.mul(nf).mul(tmin5)),
            );
        if directional.finite() && directional.hi >= 0.0 {
            remainder = remainder.min(directional.hi);
        }
        if !remainder.is_finite() || remainder < 0.0 {
            return Err(SupportError::Arithmetic("support Hessian remainder"));
        }
        Ok(finite(
            center_value.add(linear).add(I::point(remainder)),
            "support Taylor enclosure",
        )?
        .hi)
    }
}

pub(crate) struct SupportUpper {
    pub(crate) upper: f64,
    pub(crate) dense_upper: f64,
    pub(crate) taylor_upper: Option<f64>,
    pub(crate) projection_zero: bool,
    /// Actual completed early stages, distinct from the selected minimum.
    pub(crate) dense_screened: bool,
    pub(crate) cauchy_screened: bool,
    pub(crate) taylor_attempted: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_n12_f64_anchor_stays_inside_subnormal_singletons() {
        let tiny = f64::from_bits(1);
        for endpoint in [tiny, -tiny] {
            let mut poll = |_: usize| Ok::<_, ()>(());
            let geometry =
                GeometryF64::new(&[endpoint], &[endpoint], 1e-5, true, &mut poll).unwrap();
            assert_eq!(geometry.center[0].to_bits(), endpoint.to_bits());
            assert!(geometry.delta[0].lo == 0.0 && geometry.delta[0].hi == 0.0);
            assert_eq!(geometry.rho2, 0.0);
        }
    }

    #[test]
    fn decoder_n12_demand_screen_still_checks_last_original_and_polls_before_authorizing() {
        let lower = [1.0; 257];
        let upper = [1.0; 257];
        let target = [0.0; 257];
        let gamma = [1.0; 257];
        let mut check = |_: usize| Ok::<_, ()>(());
        let geometry = GeometryF64::new(&lower, &upper, 1e-5, false, &mut check).unwrap();
        let mut poisoned = [0.0; 257];
        poisoned[256] = f64::NAN;
        assert!(matches!(
            geometry.upper_for(
                &poisoned,
                &target,
                &gamma,
                None,
                -1.0,
                0.0,
                SupportDemand::StrictNegative,
                &mut check
            ),
            Err(SupportError::Arithmetic(
                "support finite original direction/parameters"
            ))
        ));
        let mut scanned = 0usize;
        let mut interrupted = |count| {
            scanned += count;
            if scanned >= 256 {
                Err(())
            } else {
                Ok(())
            }
        };
        assert!(matches!(
            geometry.upper_for(
                &target,
                &target,
                &gamma,
                None,
                -1.0,
                0.0,
                SupportDemand::StrictNegative,
                &mut interrupted
            ),
            Err(SupportError::Poll(()))
        ));
        assert_eq!(scanned, 256);
        for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                geometry.upper_for(
                    &target,
                    &target,
                    &gamma,
                    None,
                    -1.0,
                    0.0,
                    SupportDemand::AtMost(threshold),
                    &mut check
                ),
                Err(SupportError::Arithmetic("support finite replay threshold"))
            ));
        }
    }
}
