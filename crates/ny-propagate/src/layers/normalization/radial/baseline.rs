// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Normalization-only checked replay of the primitive CROWN baseline.
//! Every proposed slope is a finite real binary64 number. Endpoint and
//! critical-point evaluation authorize its intercept; interval composition
//! and signed-box residuals authorize all coefficient storage. Plane selection
//! is a heuristic, while all four McCormick planes remain valid choices.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::{checked_shape_product, Result};
use ny_tensor::BoundedTensor;

use super::primitives::{clip, compose, envelope, mccormick, Primitive};
use super::{
    append_zero_row, checkpoint, error_charge, error_plane_for_reshape, logical_error_row,
    lower_cast, refusal, sqrt_interval, sqrt_point, square, upper_cast, Geometry, I,
    MAX_CONSTRUCTION_BYTES,
};
use crate::layers::normalization::decomposed::{DecomposedNormBackwardResult, RowValidationCounts};
use crate::BatchedLinearBounds;

fn checked_row(
    c: &[f32],
    incoming_error: Option<&[f32]>,
    incoming_bias: f32,
    gamma: &[f32],
    beta: Option<&[f32]>,
    geometry: &Geometry,
    centered: bool,
    inv: I,
    var_eps: I,
    eps: f32,
    limit: f64,
    lower: bool,
) -> Result<(Vec<f32>, f32, bool)> {
    let n = c.len();
    if incoming_bias.is_nan() {
        return Err(refusal("baseline bias NaN"));
    }
    if !incoming_bias.is_finite() {
        return Ok((vec![0.0; n], incoming_bias, true));
    }
    let mut coefficients = vec![I::point(0.0); n];
    let mut bias = I::point(incoming_bias as f64);
    let mut aux = I::point(0.0);
    let mut fallback = I::point(incoming_bias as f64);
    for i in 0..n {
        if i % 256 == 0 {
            checkpoint(None)?;
        }
        if !c[i].is_finite() {
            return Err(refusal("baseline finite direction required"));
        }
        let weight = I::point(c[i] as f64 * gamma[i] as f64);
        let beta_term = I::point(c[i] as f64 * beta.map_or(0.0, |b| b[i]) as f64);
        bias = bias.add(beta_term);
        fallback = fallback.add(beta_term);
        let product = geometry.z_box[i].mul(inv);
        let normalized = I::new(product.lo.max(-limit), product.hi.min(limit));
        if !normalized.finite() {
            return Err(refusal("conditional normalization output range is empty"));
        }
        fallback = fallback.add(weight.mul(normalized));
        if weight.lo == 0.0 && weight.hi == 0.0 {
            continue;
        }
        let (clipped_weight, clip_bias) = compose(weight, clip(product, limit)?, lower)?;
        bias = bias.add(clip_bias);
        let (primary, auxiliary, constant) =
            mccormick(clipped_weight, geometry.z_box[i], inv, lower)?;
        coefficients[i] = coefficients[i].add(primary);
        aux = aux.add(auxiliary);
        bias = bias.add(constant);
    }
    let rms = sqrt_interval(var_eps).ok_or_else(|| refusal("baseline denominator range"))?;
    let (aux_rms, recip_bias) = compose(aux, envelope(Primitive::Reciprocal, rms)?, lower)?;
    bias = bias.add(recip_bias);
    let (aux_var, sqrt_bias) = compose(aux_rms, envelope(Primitive::Sqrt, var_eps)?, lower)?;
    bias = bias.add(sqrt_bias).add(aux_var.mul(I::point(eps as f64)));
    // var_eps = eps + (1/n) sum(z_i²). The caller supplies exact authored eps.
    let square_weight = aux_var.div_positive(I::point(n as f64));
    for i in 0..n {
        if i % 256 == 0 {
            checkpoint(None)?;
        }
        let (square_coefficient, square_bias) = compose(
            square_weight,
            envelope(Primitive::Square, geometry.z_box[i])?,
            lower,
        )?;
        coefficients[i] = coefficients[i].add(square_coefficient);
        bias = bias.add(square_bias);
    }
    if centered {
        let mean = coefficients
            .iter()
            .fold(I::point(0.0), |s, &c| s.add(c))
            .div_positive(I::point(n as f64));
        for c in &mut coefficients {
            *c = c.sub(mean);
        }
    }
    let mut stored = Vec::with_capacity(n);
    let mut rounding = I::point(0.0);
    let mut concrete = I::point(0.0);
    for i in 0..n {
        if i % 256 == 0 {
            checkpoint(None)?;
        }
        let coefficient = coefficients[i].midpoint() as f32;
        if !coefficients[i].finite() || !coefficient.is_finite() {
            return Err(refusal("baseline coefficient storage"));
        }
        let input_abs = I::point(geometry.center[i].abs())
            .add(I::point(geometry.radius[i]))
            .hi;
        rounding = rounding.add(
            I::point(coefficients[i].sub(I::point(coefficient as f64)).abs_max())
                .mul(I::point(input_abs)),
        );
        stored.push(coefficient);
        let input = I::new(
            I::point(geometry.center[i])
                .sub(I::point(geometry.radius[i]))
                .lo,
            I::point(geometry.center[i])
                .add(I::point(geometry.radius[i]))
                .hi,
        );
        concrete = concrete.add(I::point(coefficient as f64).mul(input));
    }
    let error = I::point(error_charge(incoming_error, gamma, beta, geometry)?);
    bias = if lower {
        bias.sub(rounding).sub(error)
    } else {
        bias.add(rounding).add(error)
    };
    fallback = if lower {
        fallback.sub(error)
    } else {
        fallback.add(error)
    };
    let mut stored_bias = if lower {
        lower_cast(bias.lo)
    } else {
        upper_cast(bias.hi)
    };
    concrete = concrete.add(I::point(stored_bias as f64));
    let fell_back = (lower && concrete.lo < fallback.lo) || (!lower && concrete.hi > fallback.hi);
    if fell_back {
        stored.fill(0.0);
        stored_bias = if lower {
            lower_cast(fallback.lo)
        } else {
            upper_cast(fallback.hi)
        };
    }
    Ok((stored, stored_bias, fell_back))
}

pub(crate) fn checked_baseline(
    bounds: &BatchedLinearBounds,
    gamma: &Array1<f32>,
    beta: Option<&Array1<f32>>,
    eps: f32,
    pre: &BoundedTensor,
    centered: bool,
    windows: Option<&[Option<(f32, f32)>]>,
) -> Result<DecomposedNormBackwardResult> {
    checkpoint(None)?;
    super::require_environment()?;
    bounds.validate_internal_shapes()?;
    let shape = bounds.lower_a.shape();
    let n = shape[shape.len() - 1];
    let outputs = shape[shape.len() - 2];
    let groups = checked_shape_product(&shape[..shape.len() - 2])
        .ok_or_else(|| refusal("baseline batch overflow"))?;
    if n == 0
        || groups == 0
        || n > (1usize << 24)
        || gamma.len() != n
        || beta.is_some_and(|b| b.len() != n)
        || !eps.is_finite()
        || eps < super::super::NORMALIZATION_MIN_EPS
    {
        return Err(refusal("checked baseline shape/epsilon"));
    }
    if pre.shape().last() != Some(&n)
        || pre.len()
            != groups
                .checked_mul(n)
                .ok_or_else(|| refusal("baseline input overflow"))?
    {
        return Err(refusal("baseline input groups"));
    }
    let bytes = bounds
        .lower_a
        .len()
        .checked_mul(24)
        .and_then(|v| n.checked_mul(512).and_then(|k| v.checked_add(k)))
        .and_then(|v| pre.len().checked_mul(8).and_then(|k| v.checked_add(k)))
        .and_then(|v| v.checked_add(16384))
        .and_then(|v| {
            super::demand::mask_bytes(groups, outputs)
                .ok()
                .and_then(|mask| v.checked_add(mask))
        })
        .ok_or_else(|| refusal("baseline memory prediction"))?;
    if bytes > MAX_CONSTRUCTION_BYTES {
        return Err(refusal("baseline memory admission"));
    }
    let demand = super::demand::inspect(bounds, pre, gamma, beta, groups, outputs, n, false, None)?;
    let lower: Vec<f32> = pre.lower().iter().copied().collect();
    let upper: Vec<f32> = pre.upper().iter().copied().collect();
    let gamma: Vec<f32> = gamma.iter().copied().collect();
    let beta = beta.map(|b| b.iter().copied().collect::<Vec<_>>());
    if gamma
        .iter()
        .chain(beta.iter().flat_map(|b| b.iter()))
        .any(|x| !x.is_finite())
    {
        return Err(refusal("baseline finite scale/shift"));
    }
    let cl = bounds
        .lower_a
        .view()
        .into_shape_with_order((groups, outputs, n))
        .map_err(|_| refusal("baseline lower layout"))?;
    let cu = bounds
        .upper_a
        .view()
        .into_shape_with_order((groups, outputs, n))
        .map_err(|_| refusal("baseline upper layout"))?;
    let bl = bounds
        .lower_b
        .view()
        .into_shape_with_order((groups, outputs))
        .map_err(|_| refusal("baseline lower bias layout"))?;
    let bu = bounds
        .upper_b
        .view()
        .into_shape_with_order((groups, outputs))
        .map_err(|_| refusal("baseline upper bias layout"))?;
    let error_lower = bounds
        .lower_a_err
        .as_ref()
        .map(|e| error_plane_for_reshape(e, None))
        .transpose()?;
    let error_upper = bounds
        .upper_a_err
        .as_ref()
        .map(|e| error_plane_for_reshape(e, None))
        .transpose()?;
    let el = error_lower
        .as_ref()
        .map(|e| e.view().into_shape_with_order((groups, outputs, n)))
        .transpose()
        .map_err(|_| refusal("baseline lower error layout"))?;
    let eu = error_upper
        .as_ref()
        .map(|e| e.view().into_shape_with_order((groups, outputs, n)))
        .transpose()
        .map_err(|_| refusal("baseline upper error layout"))?;
    let mut al = Vec::with_capacity(bounds.lower_a.len());
    let mut au = Vec::with_capacity(bounds.upper_a.len());
    let mut lb = Vec::with_capacity(bounds.lower_b.len());
    let mut ub = Vec::with_capacity(bounds.upper_b.len());
    let limit = sqrt_point(if centered { (n - 1) as f64 } else { n as f64 })
        .ok_or_else(|| refusal("baseline global norm range"))?
        .hi;
    let mut fallback_rows = 0usize;
    for group in 0..groups {
        checkpoint(None)?;
        if demand.zero_groups[group] {
            for output in 0..outputs {
                if output % 16 == 0 {
                    checkpoint(None)?;
                }
                append_zero_row(&mut al, n, None)?;
                append_zero_row(&mut au, n, None)?;
                lb.push(bl[[group, output]]);
                ub.push(bu[[group, output]]);
            }
            continue;
        }
        let geometry = Geometry::new(
            &lower[group * n..(group + 1) * n],
            &upper[group * n..(group + 1) * n],
            eps,
            centered,
            None,
        )?;
        let variance = geometry
            .z_box
            .iter()
            .fold(I::point(0.0), |s, &z| s.add(square(z)))
            .div_positive(I::point(n as f64));
        let mut var_eps = variance.add(I::point(eps as f64));
        var_eps.lo = var_eps.lo.max(eps as f64);
        let rms = sqrt_interval(var_eps).ok_or_else(|| refusal("baseline rms range"))?;
        let mut inv = I::point(1.0).div_positive(rms);
        if let Some((lo, hi)) = windows.and_then(|w| w.get(group)).copied().flatten() {
            let l = inv.lo.max(lo as f64);
            let u = inv.hi.min(hi as f64);
            if l.is_finite() && u.is_finite() && l > 0.0 && l <= u {
                inv = I::new(l, u);
                var_eps = square(I::point(1.0).div_positive(inv));
            }
        }
        for output in 0..outputs {
            if output % 16 == 0 {
                checkpoint(None)?;
            }
            if demand.zero_rows[group * outputs + output] {
                append_zero_row(&mut al, n, None)?;
                append_zero_row(&mut au, n, None)?;
                lb.push(bl[[group, output]]);
                ub.push(bu[[group, output]]);
                continue;
            }
            let rl = cl.slice(ndarray::s![group, output, ..]);
            let ru = cu.slice(ndarray::s![group, output, ..]);
            let erl = logical_error_row(
                el.as_ref().map(|e| e.slice(ndarray::s![group, output, ..])),
                None,
            )?;
            let eru = logical_error_row(
                eu.as_ref().map(|e| e.slice(ndarray::s![group, output, ..])),
                None,
            )?;
            let (a, b, lower_fallback) = checked_row(
                rl.as_slice()
                    .ok_or_else(|| refusal("baseline lower contiguous"))?,
                erl.as_deref(),
                bl[[group, output]],
                &gamma,
                beta.as_deref(),
                &geometry,
                centered,
                inv,
                var_eps,
                eps,
                limit,
                true,
            )?;
            al.extend(a);
            lb.push(b);
            let (a, b, upper_fallback) = checked_row(
                ru.as_slice()
                    .ok_or_else(|| refusal("baseline upper contiguous"))?,
                eru.as_deref(),
                bu[[group, output]],
                &gamma,
                beta.as_deref(),
                &geometry,
                centered,
                inv,
                var_eps,
                eps,
                limit,
                false,
            )?;
            au.extend(a);
            ub.push(b);
            fallback_rows += usize::from(lower_fallback || upper_fallback);
        }
    }
    checkpoint(None)?;
    let bounds = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(shape), al).map_err(|_| refusal("baseline lower shape"))?,
        ArrayD::from_shape_vec(bounds.lower_b.raw_dim(), lb)
            .map_err(|_| refusal("baseline lower bias shape"))?,
        ArrayD::from_shape_vec(IxDyn(shape), au).map_err(|_| refusal("baseline upper shape"))?,
        ArrayD::from_shape_vec(bounds.upper_b.raw_dim(), ub)
            .map_err(|_| refusal("baseline upper bias shape"))?,
        bounds.input_shape.clone(),
        bounds.output_shape.clone(),
    )?;
    checkpoint(None)?;
    Ok(DecomposedNormBackwardResult {
        bounds,
        validation: RowValidationCounts {
            fallback_rows,
            total_rows: groups * outputs,
        },
    })
}
