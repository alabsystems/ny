// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Direct scalar Taylor enclosure and stored-gradient residual charges.

use super::geometry::{norm, project};
use super::{checkpoint, refusal, Geometry, I};
use crate::certify::work::WorkBudget;
use ny_core::Result;

pub(super) struct Row {
    pub(super) a: Vec<f32>,
    pub(super) bias: I,
}

pub(super) fn row(
    c: &[f32],
    error: Option<&[f32]>,
    bias: f32,
    gamma: &[f32],
    beta: Option<&[f32]>,
    centered: bool,
    geometry: &Geometry,
    budget: Option<&WorkBudget>,
) -> Result<Row> {
    let n = c.len();
    if !bias.is_finite() || c.iter().any(|x| !x.is_finite()) {
        return Err(refusal("finite direction and bias required"));
    }
    // A product of two finite binary32 values is exactly representable in
    // binary64 (at most 48 significant bits, exponent well within its range).
    let transformed: Vec<I> = c
        .iter()
        .zip(gamma)
        .map(|(&c, &g)| I::point(c as f64 * g as f64))
        .collect();
    let a = project(&transformed, centered, budget)?;
    let mut dot = I::point(0.0);
    let mut beta_dot = I::point(bias as f64);
    let mut incoming_error = I::point(0.0);
    for i in 0..n {
        if i % 256 == 0 {
            checkpoint(budget)?;
        }
        dot = dot.add(a[i].mul(geometry.z0[i]));
        if let Some(beta) = beta {
            beta_dot = beta_dot.add(I::point(c[i] as f64 * beta[i] as f64));
        }
        if let Some(error) = error {
            if !error[i].is_finite() || error[i] < 0.0 {
                return Err(refusal("invalid incoming coefficient error"));
            }
            incoming_error = incoming_error.add(I::point(error[i] as f64).mul(I::point(
                geometry.output_abs(gamma[i], beta.map_or(0.0, |b| b[i]), i),
            )));
        }
    }
    let center_value = dot.div_positive(geometry.denominator).add(beta_dot);
    let t3 = geometry
        .denominator
        .mul(geometry.denominator)
        .mul(geometry.denominator);
    let nf = I::point(n as f64);
    let radial_factor = dot.div_positive(nf.mul(t3));
    let mut stored = Vec::with_capacity(n);
    let mut recentered = center_value;
    let mut numeric = incoming_error;
    for i in 0..n {
        if i % 256 == 0 {
            checkpoint(budget)?;
        }
        let gradient = a[i]
            .div_positive(geometry.denominator)
            .sub(radial_factor.mul(geometry.z0[i]));
        let coefficient = gradient.midpoint() as f32;
        if !gradient.finite() || !coefficient.is_finite() {
            return Err(refusal("gradient storage refused"));
        }
        stored.push(coefficient);
        recentered = recentered.sub(I::point(coefficient as f64).mul(I::point(geometry.center[i])));
        numeric = numeric.add(
            I::point(gradient.sub(I::point(coefficient as f64)).abs_max())
                .mul(I::point(geometry.radius[i])),
        );
    }
    let a_norm = norm(&a, budget)?;
    let tmin = I::point(geometry.t_min);
    let tmin2 = tmin.mul(tmin);
    let mut remainder = I::point(3.0)
        .mul(I::point(a_norm.hi))
        .mul(I::point(geometry.rho2))
        .div_positive(I::point(geometry.root_n.lo).mul(tmin2))
        .hi;
    // Directional tube bound from the exact Hessian, independent of the
    // preceding global norm bound. Both bounds cover every point of the tube.
    let mut a_v = I::point(0.0);
    let mut z0_v = I::point(0.0);
    let mut a_z_box = I::point(0.0);
    for i in 0..n {
        if i % 256 == 0 {
            checkpoint(budget)?;
        }
        a_v = a_v.add(I::point(a[i].abs_max()).mul(I::point(geometry.radius[i])));
        z0_v = z0_v.add(I::point(geometry.z0[i].abs_max()).mul(I::point(geometry.radius[i])));
        a_z_box = a_z_box.add(a[i].mul(geometry.z_box[i]));
    }
    let z_v = z0_v.add(I::point(geometry.rho2));
    let max_a_z = I::point(a_z_box.abs_max());
    let tmin3 = tmin2.mul(tmin);
    let tmin5 = tmin3.mul(tmin2);
    let directional = a_v
        .mul(z_v)
        .div_positive(nf.mul(tmin3))
        .add(
            I::point(0.5)
                .mul(max_a_z)
                .mul(I::point(geometry.rho2))
                .div_positive(nf.mul(tmin3)),
        )
        .add(
            I::point(1.5)
                .mul(max_a_z)
                .mul(z_v)
                .mul(z_v)
                .div_positive(nf.mul(nf).mul(tmin5)),
        )
        .hi;
    if directional.is_finite() && directional >= 0.0 {
        remainder = remainder.min(directional);
    }
    let charge = numeric.add(I::point(remainder));
    let result = I::new(recentered.sub(charge).lo, recentered.add(charge).hi);
    if !result.finite() {
        return Err(refusal("Taylor enclosure overflow"));
    }
    Ok(Row {
        a: stored,
        bias: result,
    })
}

pub(super) fn error_charge(
    error: Option<&[f32]>,
    gamma: &[f32],
    beta: Option<&[f32]>,
    geometry: &Geometry,
) -> Result<f64> {
    let mut charge = I::point(0.0);
    if let Some(error) = error {
        for (i, &e) in error.iter().enumerate() {
            if !e.is_finite() || e < 0.0 {
                return Err(refusal("invalid incoming coefficient error"));
            }
            charge = charge.add(I::point(e as f64).mul(I::point(geometry.output_abs(
                gamma[i],
                beta.map_or(0.0, |b| b[i]),
                i,
            ))));
        }
    }
    Ok(charge.hi)
}
