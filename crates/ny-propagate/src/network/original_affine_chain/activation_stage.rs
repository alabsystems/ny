// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sealed exact-real GELU-erf/SiLU facts for an original unary affine chain.
//! This adapter uses only checked scalar kernels, never generic evaluators.
//! Every fact/output remains attached to the caller's original phase claim.

use ndarray::{Array1, Array2, ArrayD, IxDyn};

use crate::layers::linear::bias::{
    add_f64_down, add_f64_up, f32_to_f64_exact, f64_to_f32_down_no_subnormal,
    f64_to_f32_up_no_subnormal, nonnegative_f32_error_or_infinity, publish_error_up_normal,
};

use super::activation::{CheckedActivation, MathematicalActivation, ScalarEnvelope};
use super::*;

#[cfg(test)]
thread_local! {
    static INPUT_CANCEL: std::cell::RefCell<Option<(usize, crate::certify::residency::CancellationToken)>> = const { std::cell::RefCell::new(None) };
}

pub(in crate::network) struct OwnedActivationFact(RetainedOwner<ActivationFactData>);

pub(in crate::network) struct ActivationFactData {
    bounds: BoundedTensor,
    envelopes: Vec<ScalarEnvelope>,
    input: Digest,
    output: Digest,
    budget: crate::certify::work::WorkBudget,
    deadline: Option<Instant>,
}
impl std::ops::Deref for OwnedActivationFact {
    type Target = ActivationFactData;

    fn deref(&self) -> &ActivationFactData {
        &self.0
    }
}
impl OwnedActivationFact {
    pub(super) fn bounds(&self) -> &BoundedTensor {
        &self.bounds
    }
    pub(super) fn reserved_bytes(&self) -> usize {
        self.0.bytes()
    }
}

fn output_vec(len: usize, phase: &mut OriginalAffinePhase) -> Result<Vec<f32>> {
    let admitted = add(mul(len, size_of::<f32>())?, 128)?;
    phase.retain_metadata(admitted)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| invalid("activation output allocation"))?;
    if mul(values.capacity(), size_of::<f32>())? > admitted {
        return Err(invalid("activation output capacity exceeded admission"));
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

fn forward_request(input: &BoundedTensor) -> Result<WorkRequest> {
    let count = cap(input.len())?;
    if count == 0 || input.shape().len() > 16 || input.lower().shape() != input.upper().shape() {
        return Err(invalid("nonempty activation endpoint shape required"));
    }
    RetainedOwner::<ActivationFactData>::check_metadata(META)
        .map_err(|e| invalid(&format!("activation fact metadata capacity: {e}")))?;
    let mut admitted = activation::request(count)?;
    let metadata = add(META, mul(add(input.shape().len(), 1)?, 128)?)?;
    admitted.persistent_bytes = add(
        admitted.persistent_bytes,
        add(metadata, add(mul(count, 8)?, 256)?)?,
    )?;
    // Scalar request covers every recurrence/envelope. The additional census
    // covers finite input inspection, two initialized outputs, polled tensor
    // validation, two endpoint hashes, shape/capacity metadata and final scans.
    admitted.work = admitted
        .work
        .checked_add(as_work(add(mul(count, 40)?, 4096)?)?)
        .ok_or_else(|| invalid("activation forward work overflow"))?;
    Ok(admitted)
}

pub(super) fn forward(
    input: &BoundedTensor,
    options: OriginalAffineOptions<'_>,
) -> Result<OwnedActivationFact> {
    forward_profile(input, MathematicalActivation::GeluErf, options)
}

pub(super) fn forward_profile(
    input: &BoundedTensor,
    profile: MathematicalActivation,
    options: OriginalAffineOptions<'_>,
) -> Result<OwnedActivationFact> {
    let admission = forward_request(input)?;
    let mut phase =
        OriginalAffinePhase::admit(options.budget, options.class, admission, options.deadline)?;
    phase.entry()?;
    let count = input.len();
    let lo = input
        .lower()
        .as_slice()
        .ok_or_else(|| invalid("activation requires standard original forward facts"))?;
    let hi = input
        .upper()
        .as_slice()
        .ok_or_else(|| invalid("activation requires standard original forward facts"))?;
    for i in 0..count {
        if i.is_multiple_of(POLL) {
            phase.checkpoint()?;
        }
        phase.charge_work(8)?;
        let l = f32_to_f64_exact(lo[i]);
        let u = f32_to_f64_exact(hi[i]);
        if !l.is_finite() || !u.is_finite() || l > u {
            return Err(numeric("nonfinite/inverted activation facts"));
        }
        #[cfg(test)]
        INPUT_CANCEL.with(|hook| {
            let mut hook = hook.borrow_mut();
            if let Some((left, cancellation)) = hook.as_mut() {
                *left -= 1;
                if *left == 0 {
                    cancellation.cancel();
                    *hook = None;
                }
            }
        });
        #[cfg(test)]
        phase.checkpoint()?;
    }
    let metadata = add(META, mul(add(input.shape().len(), 1)?, 128)?)?;
    phase.retain_metadata(metadata)?;
    let input_digest = hash_interval(input, &mut phase)?;
    // This phase admitted all requested envelope capacity before allocation.
    // Each private enclose retains its exact envelope once. Fixed metadata
    // covers the Vec header and bounded allocator capacity slack, not a new
    // authority for the already priced N scalar payloads.
    let mut envelopes = Vec::new();
    envelopes
        .try_reserve_exact(count)
        .map_err(|_| invalid("activation envelope allocation"))?;
    if mul(envelopes.capacity(), size_of::<ScalarEnvelope>())?
        > add(mul(count, size_of::<ScalarEnvelope>())?, 128)?
    {
        return Err(invalid("activation envelope capacity exceeded admission"));
    }
    let mut lower = output_vec(count, &mut phase)?;
    let mut upper = output_vec(count, &mut phase)?;
    {
        let kernel = CheckedActivation::prepare(profile, &mut phase)?;
        for i in 0..count {
            phase.checkpoint()?;
            let envelope = kernel.enclose(lo[i], hi[i], &mut phase)?;
            (lower[i], upper[i]) = envelope.image();
            phase.charge_work(4)?;
            envelopes.push(envelope);
        }
    }
    // Arithmetic constants never cross into a backward/query phase. Only the
    // checked, immutable output lines/boxes remain under this paid fact owner.
    let shape = IxDyn(input.shape());
    let lower = ArrayD::from_shape_vec(shape.clone(), lower)
        .map_err(|_| invalid("activation lower reshape"))?;
    let upper =
        ArrayD::from_shape_vec(shape, upper).map_err(|_| invalid("activation upper reshape"))?;
    phase.charge_work(as_work(mul(count, 4)?)?)?;
    let bounds = BoundedTensor::new_allow_infinite_with_poll(lower, upper, || phase.checkpoint())?;
    let output_digest = hash_interval(&bounds, &mut phase)?;
    phase.checkpoint()?;
    Ok(OwnedActivationFact(phase.commit_retained_with(|| {
        ActivationFactData {
            bounds,
            envelopes,
            input: input_digest,
            output: output_digest,
            budget: options.budget.clone(),
            deadline: earlier(options.deadline, ny_core::instance_budget::deadline()),
        }
    })?))
}

fn backward_request(rows: usize, columns: usize) -> Result<WorkRequest> {
    let elements = cap(mul(rows, columns)?)?;
    let count = add(mul(elements, 4)?, mul(rows, 2)?)?;
    // 256 terms/cell covers both signs, both input error planes, initialization
    // and validation of all six outputs, exact f32 products/storage gaps and
    // directed bias sums. 96/column covers fact/input/output binding scans
    // and the 32-credit construction metadata check for every retained scalar.
    request(
        add(
            add(mul(elements, 256)?, mul(columns, 96)?)?,
            add(mul(rows, 32)?, 4096)?,
        )?,
        add(META, add(mul(count, 4)?, 6 * 128)?)?,
    )
}

fn validate_fact(
    fact: &OwnedActivationFact,
    pre: &BoundedTensor,
    profile: MathematicalActivation,
    phase: &mut OriginalAffinePhase,
) -> Result<()> {
    if pre.shape().len() > 16
        || pre.lower().shape() != pre.upper().shape()
        || pre.shape() != fact.bounds.shape()
        || pre.len() != fact.envelopes.len()
        || fact.bounds.lower().shape() != fact.bounds.upper().shape()
    {
        return Err(invalid("activation original fact/input shape mismatch"));
    }
    if hash_interval(pre, phase)? != fact.input
        || hash_interval(&fact.bounds, phase)? != fact.output
    {
        return Err(invalid("activation fact original box binding mismatch"));
    }
    let lo = pre
        .lower()
        .as_slice()
        .ok_or_else(|| invalid("activation standard preactivation facts required"))?;
    let hi = pre
        .upper()
        .as_slice()
        .ok_or_else(|| invalid("activation standard preactivation facts required"))?;
    let image_lo = fact
        .bounds
        .lower()
        .as_slice()
        .ok_or_else(|| invalid("activation image layout"))?;
    let image_hi = fact
        .bounds
        .upper()
        .as_slice()
        .ok_or_else(|| invalid("activation image layout"))?;
    for (i, envelope) in fact.envelopes.iter().enumerate() {
        if i.is_multiple_of(POLL) {
            phase.checkpoint()?;
        }
        phase.charge_work(32)?;
        let (l, u) = envelope.interval();
        let (image_l, image_u) = envelope.image();
        let (b_l, b_u) = envelope.intercepts();
        if envelope.profile() != profile
            || l.to_bits() != lo[i].to_bits()
            || u.to_bits() != hi[i].to_bits()
            || image_l.to_bits() != image_lo[i].to_bits()
            || image_u.to_bits() != image_hi[i].to_bits()
            || [l, u, image_l, image_u, envelope.slope(), b_l, b_u]
                .iter()
                .any(|&v| !f32_to_f64_exact(v).is_finite())
            || f32_to_f64_exact(l) > f32_to_f64_exact(u)
            || f32_to_f64_exact(image_l) > f32_to_f64_exact(image_u)
            || f32_to_f64_exact(b_l) > f32_to_f64_exact(b_u)
        {
            return Err(numeric("invalid checked activation fact"));
        }
        envelope.validate_construction_metadata(phase)?;
    }
    phase.checkpoint()
}

fn stored_product(value: f64, slope: f32) -> Result<(f32, f32)> {
    // Both original operands are binary32 bit values: their product is exact
    // in binary64. Only its publication to binary32 needs an error plane.
    let product = value * f32_to_f64_exact(slope);
    let mut stored = product as f32;
    if !stored.is_finite() {
        return Err(numeric("activation coefficient storage overflow"));
    }
    if stored.to_bits() & 0x7f80_0000 == 0 {
        stored = 0.0;
    }
    let exact_stored = f32_to_f64_exact(stored);
    let error = if exact_stored == product {
        0.0
    } else {
        publish_error_up_normal(add_f64_up(
            exact_stored.max(product),
            -exact_stored.min(product),
        ))
    };
    if !error.is_finite() {
        return Err(numeric("activation coefficient-error storage overflow"));
    }
    Ok((stored, error))
}

fn coefficient(
    nominal: f32,
    incoming_error: f32,
    envelope: &ScalarEnvelope,
    lower: bool,
    bias: &mut f64,
) -> Result<(f32, f32)> {
    let value = f32_to_f64_exact(nominal);
    let error = nonnegative_f32_error_or_infinity(incoming_error);
    if !value.is_finite() || !error.is_finite() {
        return Err(numeric("invalid activation incoming coefficient/error"));
    }
    let (image_l, image_u) = envelope.image();
    let magnitude = f32_to_f64_exact(image_l)
        .abs()
        .max(f32_to_f64_exact(image_u).abs());
    // Discharge complete uncertain coefficient support BEFORE choosing a
    // nominal sign. This includes error-only and sign-straddling carriers.
    let charge = error * magnitude; // exact binary32 product in binary64
    *bias = if lower {
        add_f64_down(*bias, -charge)
    } else {
        add_f64_up(*bias, charge)
    };
    let (lower_b, upper_b) = envelope.intercepts();
    let intercept = if lower == (value >= 0.0) {
        lower_b
    } else {
        upper_b
    };
    let contribution = value * f32_to_f64_exact(intercept);
    *bias = if lower {
        add_f64_down(*bias, contribution)
    } else {
        add_f64_up(*bias, contribution)
    };
    stored_product(value, envelope.slope())
}

pub(super) fn backward(
    incoming: &LinearBounds,
    pre: &BoundedTensor,
    fact: &OwnedActivationFact,
    options: OriginalAffineOptions<'_>,
) -> Result<OwnedRows> {
    backward_profile(
        incoming,
        pre,
        fact,
        MathematicalActivation::GeluErf,
        options,
    )
}

pub(super) fn backward_profile(
    incoming: &LinearBounds,
    pre: &BoundedTensor,
    fact: &OwnedActivationFact,
    profile: MathematicalActivation,
    options: OriginalAffineOptions<'_>,
) -> Result<OwnedRows> {
    if !fact.budget.shares_ledger(options.budget) {
        return Err(NyError::UnsupportedConfiguration(
            "original activation fact/caller ledger mismatch".into(),
        ));
    }
    let (rows, columns) = linear_shape(incoming)?;
    if pre.len() != columns || fact.envelopes.len() != columns {
        return Err(invalid("activation backward position/feature shape"));
    }
    let admission = backward_request(rows, columns)?;
    let mut phase = OriginalAffinePhase::admit(
        options.budget,
        options.class,
        admission,
        earlier(options.deadline, fact.deadline),
    )?;
    phase.entry()?;
    validate_fact(fact, pre, profile, &mut phase)?;
    for (i, &value) in incoming
        .lower_a()
        .iter()
        .chain(incoming.upper_a().iter())
        .chain(incoming.lower_b().iter())
        .chain(incoming.upper_b().iter())
        .enumerate()
    {
        if i.is_multiple_of(POLL) {
            phase.checkpoint()?;
        }
        phase.charge_work(4)?;
        if !f32_to_f64_exact(value).is_finite() {
            return Err(numeric("nonfinite activation incoming carrier"));
        }
    }
    for error in [incoming.lower_a_err(), incoming.upper_a_err()]
        .into_iter()
        .flatten()
    {
        for (i, &value) in error.iter().enumerate() {
            if i.is_multiple_of(POLL) {
                phase.checkpoint()?;
            }
            phase.charge_work(4)?;
            if !nonnegative_f32_error_or_infinity(value).is_finite() {
                return Err(numeric("poisoned activation incoming coefficient error"));
            }
        }
    }
    phase.retain_metadata(META)?;
    let elements = rows * columns; // checked before admission
    let mut la = output_vec(elements, &mut phase)?;
    let mut ua = output_vec(elements, &mut phase)?;
    let mut le = output_vec(elements, &mut phase)?;
    let mut ue = output_vec(elements, &mut phase)?;
    let mut lb = output_vec(rows, &mut phase)?;
    let mut ub = output_vec(rows, &mut phase)?;
    for row in 0..rows {
        phase.checkpoint()?;
        let mut lower_bias = f32_to_f64_exact(incoming.lower_b()[row]);
        let mut upper_bias = f32_to_f64_exact(incoming.upper_b()[row]);
        for column in 0..columns {
            if column.is_multiple_of(POLL) {
                phase.checkpoint()?;
            }
            phase.charge_work(160)?;
            let index = row * columns + column;
            (la[index], le[index]) = coefficient(
                incoming.lower_a()[[row, column]],
                incoming.lower_a_err().map_or(0.0, |e| e[[row, column]]),
                &fact.envelopes[column],
                true,
                &mut lower_bias,
            )?;
            (ua[index], ue[index]) = coefficient(
                incoming.upper_a()[[row, column]],
                incoming.upper_a_err().map_or(0.0, |e| e[[row, column]]),
                &fact.envelopes[column],
                false,
                &mut upper_bias,
            )?;
        }
        phase.charge_work(16)?;
        lb[row] = f64_to_f32_down_no_subnormal(lower_bias);
        ub[row] = f64_to_f32_up_no_subnormal(upper_bias);
        if !lb[row].is_finite() || !ub[row].is_finite() {
            return Err(numeric("activation bias storage overflow"));
        }
    }
    let matrix = |values| {
        Array2::from_shape_vec((rows, columns), values)
            .map_err(|_| invalid("activation carrier reshape"))
    };
    let bounds = LinearBounds::from_prevalidated_parts_with_optional_err(
        matrix(la)?,
        Array1::from_vec(lb),
        matrix(ua)?,
        Array1::from_vec(ub),
        Some(matrix(le)?),
        Some(matrix(ue)?),
    )?;
    phase.checkpoint()?;
    let reservation = phase.commit()?;
    Ok(OwnedRows {
        bounds,
        _reservation: reservation,
    })
}

#[cfg(test)]
mod tests;
