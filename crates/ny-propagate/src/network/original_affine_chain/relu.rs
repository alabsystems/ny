// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::dd::next_up_f64;

use crate::layers::linear::bias::{
    add_f64_down, add_f64_up, f32_to_f64_exact, f64_to_f32_down_no_subnormal,
    f64_to_f32_up_no_subnormal, nonnegative_f32_error_or_infinity, publish_error_up_normal,
};

use super::*;

fn output_vec(len: usize, phase: &mut OriginalAffinePhase) -> Result<Vec<f32>> {
    let admitted = add(mul(len, size_of::<f32>())?, 128)?;
    phase.retain_metadata(admitted)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| invalid("ReLU output allocation"))?;
    if mul(values.capacity(), size_of::<f32>())? > admitted {
        return Err(invalid("ReLU output capacity exceeded admission"));
    }
    while values.len() < len {
        phase.checkpoint()?;
        let amount = (len - values.len()).min(POLL);
        phase.charge_work(as_work(amount)?)?;
        values.resize(values.len() + amount, 0.0);
    }
    phase.checkpoint()?;
    Ok(values)
}

pub(super) fn forward(
    input: &BoundedTensor,
    options: OriginalAffineOptions<'_>,
) -> Result<OwnedBox> {
    let count = cap(input.len())?;
    if input.lower().shape() != input.upper().shape() || input.shape().len() > 16 {
        return Err(invalid("ReLU endpoint shape"));
    }
    let metadata = add(META, mul(input.shape().len(), 64)?)?;
    RetainedOwner::<OwnedBoxData>::check_metadata(META)
        .map_err(|e| invalid(&format!("ReLU fact metadata capacity: {e}")))?;
    let admission = request(mul(count, 32)?, add(metadata, add(mul(count, 8)?, 256)?)?)?;
    let mut phase =
        OriginalAffinePhase::admit(options.budget, options.class, admission, options.deadline)?;
    phase.retain_metadata(metadata)?;
    let lo = input
        .lower()
        .as_slice()
        .ok_or_else(|| invalid("ReLU requires standard forward facts"))?;
    let hi = input
        .upper()
        .as_slice()
        .ok_or_else(|| invalid("ReLU requires standard forward facts"))?;
    for (i, (&l, &u)) in lo.iter().zip(hi).enumerate() {
        if i.is_multiple_of(POLL) {
            phase.checkpoint()?;
        }
        phase.charge_work(8)?;
        if !l.is_finite() || !u.is_finite() || f32_to_f64_exact(l) > f32_to_f64_exact(u) {
            return Err(numeric("nonfinite/inverted ReLU facts"));
        }
    }
    let mut lower = output_vec(count, &mut phase)?;
    let mut upper = output_vec(count, &mut phase)?;
    for i in 0..count {
        if i.is_multiple_of(POLL) {
            phase.checkpoint()?;
        }
        phase.charge_work(4)?;
        // Bit-decoded comparison preserves positive subnormals even on DAZ.
        lower[i] = if f32_to_f64_exact(lo[i]) > 0.0 {
            lo[i]
        } else {
            0.0
        };
        upper[i] = if f32_to_f64_exact(hi[i]) > 0.0 {
            hi[i]
        } else {
            0.0
        };
    }
    let shape = IxDyn(input.shape());
    let lower =
        ArrayD::from_shape_vec(shape.clone(), lower).map_err(|_| invalid("ReLU lower reshape"))?;
    let upper = ArrayD::from_shape_vec(shape, upper).map_err(|_| invalid("ReLU upper reshape"))?;
    phase.charge_work(as_work(mul(count, 4)?)?)?;
    let bounds = BoundedTensor::new_allow_infinite_with_poll(lower, upper, || phase.checkpoint())?;
    Ok(OwnedBox(
        phase.commit_retained_with(|| OwnedBoxData { bounds })?,
    ))
}

/// Stored positive upper chord. Its two endpoint inequalities are checked
/// independently of a nearest host cast. Decoded f32 endpoints are exact in
/// f64; outward width/division and publication cover the mathematical chord.
fn chord(l: f64, u: f64) -> Result<(f32, f32)> {
    let width_lo = add_f64_down(u, -l);
    if !width_lo.is_finite() || width_lo <= 0.0 {
        return Err(numeric("ReLU chord width"));
    }
    let slope = f64_to_f32_up_no_subnormal(next_up_f64(u / width_lo));
    let s = f32_to_f64_exact(slope);
    let intercept = f64_to_f32_up_no_subnormal(-(s * l));
    let b = f32_to_f64_exact(intercept);
    // The f32*f32 product is exact in f64. Directed sums check both endpoints;
    // the resulting affine line therefore encloses the convex ReLU graph.
    if !slope.is_finite()
        || !intercept.is_finite()
        || add_f64_down(s * l, b) < 0.0
        || add_f64_down(s * u, b) < u
    {
        // Directed endpoint checks can decline a line exactly touching zero.
        // Raise the intercept by one outward representable step and check it.
        let intercept = f64_to_f32_up_no_subnormal(next_up_f64(b));
        let b = f32_to_f64_exact(intercept);
        if !slope.is_finite()
            || !intercept.is_finite()
            || add_f64_down(s * l, b) < 0.0
            || add_f64_down(s * u, b) < u
        {
            return Err(numeric("ReLU upper chord endpoint check"));
        }
        return Ok((slope, intercept));
    }
    Ok((slope, intercept))
}

fn stored_product(a: f64, slope: f32) -> Result<(f32, f32)> {
    let product = a * f32_to_f64_exact(slope); // exact original f32*f32
    let mut center = product as f32;
    if !center.is_finite() {
        return Err(numeric("ReLU coefficient overflow"));
    }
    if center.to_bits() & 0x7f80_0000 == 0 {
        center = 0.0;
    }
    let exact_center = f32_to_f64_exact(center);
    let error = if exact_center == product {
        0.0
    } else {
        publish_error_up_normal(add_f64_up(
            exact_center.max(product),
            -exact_center.min(product),
        ))
    };
    if !error.is_finite() {
        return Err(numeric("ReLU coefficient error overflow"));
    }
    Ok((center, error))
}

fn coefficient(
    a: f32,
    error: f32,
    l: f32,
    u: f32,
    lower: bool,
    bias: &mut f64,
) -> Result<(f32, f32)> {
    let value = f32_to_f64_exact(a);
    let e = nonnegative_f32_error_or_infinity(error);
    let l = f32_to_f64_exact(l);
    let u = f32_to_f64_exact(u);
    if !value.is_finite() || !e.is_finite() || !l.is_finite() || !u.is_finite() || l > u {
        return Err(numeric("invalid ReLU carrier/error/facts"));
    }
    if l >= 0.0 {
        return Ok((a, error));
    }
    if u <= 0.0 {
        return Ok((0.0, 0.0));
    }
    // Crossing: discharge the complete incoming coefficient uncertainty over
    // 0<=ReLU(x)<=u. This avoids trusting the sign of an uncertain coefficient.
    let charge = e * u; // exact binary32 product
    *bias = if lower {
        add_f64_down(*bias, -charge)
    } else {
        add_f64_up(*bias, charge)
    };
    let upper_line = if lower { value < 0.0 } else { value >= 0.0 };
    let (slope, intercept) = if upper_line {
        chord(l, u)?
    } else {
        (if u > -l { 1.0 } else { 0.0 }, 0.0)
    };
    let contribution = value * f32_to_f64_exact(intercept);
    *bias = if lower {
        add_f64_down(*bias, contribution)
    } else {
        add_f64_up(*bias, contribution)
    };
    stored_product(value, slope)
}

pub(super) fn backward(
    incoming: &LinearBounds,
    pre: &BoundedTensor,
    options: OriginalAffineOptions<'_>,
) -> Result<OwnedRows> {
    let (rows, columns) = linear_shape(incoming)?;
    if pre.lower().shape() != pre.upper().shape() || pre.len() != columns {
        return Err(invalid("ReLU backward fact shape"));
    }
    let elements = cap(mul(rows, columns)?)?;
    let count = add(mul(elements, 4)?, mul(rows, 2)?)?;
    let admission = request(
        add(mul(elements, 160)?, mul(columns, 16)?)?,
        add(META, add(mul(count, 4)?, 6 * 128)?)?,
    )?;
    let mut phase =
        OriginalAffinePhase::admit(options.budget, options.class, admission, options.deadline)?;
    let lo = pre
        .lower()
        .as_slice()
        .ok_or_else(|| invalid("ReLU standard preactivation facts required"))?;
    let hi = pre
        .upper()
        .as_slice()
        .ok_or_else(|| invalid("ReLU standard preactivation facts required"))?;
    for i in 0..columns {
        if i.is_multiple_of(POLL) {
            phase.checkpoint()?;
        }
        phase.charge_work(8)?;
        if !lo[i].is_finite()
            || !hi[i].is_finite()
            || f32_to_f64_exact(lo[i]) > f32_to_f64_exact(hi[i])
        {
            return Err(numeric("invalid ReLU preactivation facts"));
        }
    }
    // All finite/error/profile checks precede the six output allocations.
    for (index, value) in incoming
        .lower_a()
        .iter()
        .chain(incoming.upper_a().iter())
        .chain(incoming.lower_b().iter())
        .chain(incoming.upper_b().iter())
        .enumerate()
    {
        if index.is_multiple_of(POLL) {
            phase.checkpoint()?;
        }
        phase.charge_work(4)?;
        if !value.is_finite() {
            return Err(numeric("nonfinite ReLU incoming carrier"));
        }
    }
    for error in [incoming.lower_a_err(), incoming.upper_a_err()]
        .into_iter()
        .flatten()
    {
        for (index, &value) in error.iter().enumerate() {
            if index.is_multiple_of(POLL) {
                phase.checkpoint()?;
            }
            phase.charge_work(4)?;
            if !nonnegative_f32_error_or_infinity(value).is_finite() {
                return Err(numeric("poisoned ReLU coefficient error"));
            }
        }
    }
    phase.retain_metadata(META)?;
    let mut la = output_vec(elements, &mut phase)?;
    let mut ua = output_vec(elements, &mut phase)?;
    let mut le = output_vec(elements, &mut phase)?;
    let mut ue = output_vec(elements, &mut phase)?;
    let mut lb = output_vec(rows, &mut phase)?;
    let mut ub = output_vec(rows, &mut phase)?;
    for row in 0..rows {
        let mut lower_bias = f32_to_f64_exact(incoming.lower_b()[row]);
        let mut upper_bias = f32_to_f64_exact(incoming.upper_b()[row]);
        for column in 0..columns {
            if column.is_multiple_of(POLL) {
                phase.checkpoint()?;
            }
            phase.charge_work(112)?;
            let index = row * columns + column;
            let e = incoming.lower_a_err().map_or(0.0, |e| e[[row, column]]);
            let f = incoming.upper_a_err().map_or(0.0, |e| e[[row, column]]);
            (la[index], le[index]) = coefficient(
                incoming.lower_a()[[row, column]],
                e,
                lo[column],
                hi[column],
                true,
                &mut lower_bias,
            )?;
            (ua[index], ue[index]) = coefficient(
                incoming.upper_a()[[row, column]],
                f,
                lo[column],
                hi[column],
                false,
                &mut upper_bias,
            )?;
        }
        phase.charge_work(16)?;
        lb[row] = f64_to_f32_down_no_subnormal(lower_bias);
        ub[row] = f64_to_f32_up_no_subnormal(upper_bias);
        if !lb[row].is_finite() || !ub[row].is_finite() {
            return Err(numeric("ReLU bias storage overflow"));
        }
    }
    let matrix = |values| {
        Array2::from_shape_vec((rows, columns), values).map_err(|_| invalid("ReLU carrier reshape"))
    };
    let bounds = LinearBounds::from_prevalidated_parts_with_optional_err(
        matrix(la)?,
        Array1::from_vec(lb),
        matrix(ua)?,
        Array1::from_vec(ub),
        Some(matrix(le)?),
        Some(matrix(ue)?),
    )?;
    let reservation = phase.commit()?;
    Ok(OwnedRows {
        bounds,
        _reservation: reservation,
    })
}
