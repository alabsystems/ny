// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cooperative storage merge for a private, leased pending transaction.
//! Numerical elements CALL the original reference's shared primitives. This
//! module admits matching layouts only; any refusal drops the entire private
//! transaction before the untouched reference fallback can run.

use super::super::super::indexed_pending::{merge_bias_elem, merged_err_elem};
use super::{
    copy_into, invalid, BoundedBatchPending, BoundedBatchSlot, BoundedRelation, Poll, Result,
    ENTRY, PENDING_COMMIT,
};
use crate::amortise::{CooperativeWork, WeightedPoll};
use crate::{GraphNetwork, LinearBounds};
use ndarray::{Array1, Array2};
use std::sync::Arc;

#[cfg(test)]
use super::BoundedLinearBounds;

const COEFFICIENT: &str = "bounded SoA merge coefficient";
const BIAS: &str = "bounded SoA merge bias";
const PRESENCE: &str = "bounded SoA merge exact marker";
const ERROR: &str = "bounded SoA merge error";
const DOMAIN_COMMIT: &str = "bounded SoA merge domain commit";

#[cfg(test)]
thread_local! {
    // Inject an expiry only AFTER a real bounded mutation stride, so rollback
    // tests cannot pass merely by refusing before work starts.
    static FAIL_AFTER_WORK: std::cell::Cell<Option<(&'static str, usize)>> = const { std::cell::Cell::new(None) };
    static OBSERVED_WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn work(poll: &mut Poll, units: usize, stage: &'static str) -> Result<()> {
    #[cfg(test)]
    {
        let fail = FAIL_AFTER_WORK.with(std::cell::Cell::get);
        if let Some((selected, after)) = fail {
            if selected == stage {
                let done = OBSERVED_WORK.with(|count| {
                    let done = count.get().saturating_add(units);
                    count.set(done);
                    done
                });
                if done >= after {
                    return Err(ny_core::NyError::DeadlineExceeded(format!(
                        "bounded SoA merge forced expiry after {done} work units at {stage}"
                    )));
                }
            }
        }
    }
    poll.work(units, stage)?;
    poll.checkpoint(stage)
}

fn layout(bounds: &LinearBounds, rows: usize, cols: usize) -> Result<()> {
    if rows == 0
        || cols == 0
        || bounds.lower_a().dim() != (rows, cols)
        || bounds.upper_a().dim() != (rows, cols)
        || bounds.lower_b().len() != rows
        || bounds.upper_b().len() != rows
        || bounds
            .lower_a_err()
            .is_some_and(|a| a.dim() != (rows, cols))
        || bounds
            .upper_a_err()
            .is_some_and(|a| a.dim() != (rows, cols))
    {
        return Err(invalid("merge contribution has an unsupported layout"));
    }
    Ok(())
}

fn coefficients(
    destination: &mut [f32],
    source: &Array2<f32>,
    lower: bool,
    poll: &mut Poll,
) -> Result<()> {
    poll.checkpoint(COEFFICIENT)?;
    poll.admit_elements(destination.len(), 64, COEFFICIENT)?;
    let price = if poll.query.is_some() { 64 } else { 1 };
    let mut chunked = WeightedPoll::new(|units| work(poll, units, COEFFICIENT));
    for (old, &new) in destination.iter_mut().zip(source.iter()) {
        chunked.before(price)?;
        *old = GraphNetwork::safe_add_elem(*old, new, lower);
    }
    chunked.checkpoint()
}

fn biases(
    destination: &mut [f32],
    source: &Array1<f32>,
    lower: bool,
    poll: &mut Poll,
) -> Result<()> {
    poll.checkpoint(BIAS)?;
    poll.admit_elements(destination.len(), 32, BIAS)?;
    let price = if poll.query.is_some() { 32 } else { 1 };
    let mut chunked = WeightedPoll::new(|units| work(poll, units, BIAS));
    for (old, &new) in destination.iter_mut().zip(source.iter()) {
        chunked.before(price)?;
        *old = merge_bias_elem(*old, new, lower);
    }
    chunked.checkpoint()
}

fn error_present(
    old_present: bool,
    source: Option<&Array2<f32>>,
    merged_coefficients: &[f32],
    poll: &mut Poll,
) -> Result<bool> {
    poll.checkpoint(PRESENCE)?;
    if old_present || source.is_some() {
        return Ok(true);
    }
    let mut present = false;
    poll.admit_elements(merged_coefficients.len(), 1, PRESENCE)?;
    let price = 1;
    let mut chunked = WeightedPoll::new(|units| work(poll, units, PRESENCE));
    for &coefficient in merged_coefficients {
        chunked.before(price)?;
        present |= coefficient != 0.0;
        if present {
            break;
        }
    }
    chunked.checkpoint()?;
    Ok(present)
}

fn errors(
    destination: &mut [f32],
    old_present: bool,
    source: Option<&Array2<f32>>,
    merged_coefficients: &[f32],
    required: bool,
    poll: &mut Poll,
) -> Result<()> {
    poll.checkpoint(ERROR)?;
    poll.admit_elements(destination.len(), 64, ERROR)?;
    let mut source = source.map(|a| a.iter());
    let price = if poll.query.is_some() { 64 } else { 1 };
    let mut chunked = WeightedPoll::new(|units| work(poll, units, ERROR));
    for (old, &coefficient) in destination.iter_mut().zip(merged_coefficients) {
        chunked.before(price)?;
        *old = if required {
            let new = source.as_mut().map(|values| {
                *values
                    .next()
                    .expect("merge source shape checked before mutation")
            });
            merged_err_elem(old_present.then_some(*old), new, coefficient)
        } else {
            // Preserve set_coeff_err's pairing: the absent side is an exact
            // zero plane when the other side requires an error plane.
            0.0
        };
    }
    chunked.checkpoint()
}

pub(super) fn accumulate(
    mut pending: BoundedBatchPending,
    idx: usize,
    domain: usize,
    contribution: &impl BoundedRelation,
    poll: &mut Poll,
) -> Result<BoundedBatchPending> {
    poll.checkpoint(ENTRY)?;
    if domain >= pending.domains || idx >= pending.slots.len() {
        return Err(invalid("merge target slot or domain is unknown"));
    }
    if !Arc::ptr_eq(&pending.charge.owner, contribution.owner()) {
        return Err(invalid(
            "merge contribution belongs to a different live account",
        ));
    }
    pending.charge.owner.check_live(poll)?;
    let bounds = contribution.bounds();
    let rows = bounds.num_outputs();
    let cols = bounds.num_inputs();
    layout(bounds, rows, cols)?;
    if let Some(slot) = &pending.slots[idx] {
        if (slot.storage.rows, slot.storage.cols) != (rows, cols) {
            return Err(invalid("merge target layout differs from its contribution"));
        }
    } else {
        pending.slots[idx] = Some(BoundedBatchSlot::zeroed_with_poll(
            &pending.charge.owner,
            pending.domains,
            rows,
            cols,
            pending.deadline,
            poll,
        )?);
    }
    let storage = &mut pending.slots[idx]
        .as_mut()
        .expect("slot staged above")
        .storage;
    // The slot was constructed and validated privately; checked allocation
    // established these products before any slice or mutation occurs.
    let rc = rows * cols;
    let coefficients_range = domain * rc..(domain + 1) * rc;
    let biases_range = domain * rows..(domain + 1) * rows;
    let la = &mut storage.lower_a[coefficients_range.clone()];
    let ua = &mut storage.upper_a[coefficients_range.clone()];
    let lb = &mut storage.lower_b[biases_range.clone()];
    let ub = &mut storage.upper_b[biases_range];
    let le = &mut storage.lower_a_err[coefficients_range.clone()];
    let ue = &mut storage.upper_a_err[coefficients_range];
    if !storage.active[domain] {
        for (destination, source) in [(la, bounds.lower_a()), (ua, bounds.upper_a())] {
            copy_into(destination, source.as_slice(), source.iter().copied(), poll)?;
        }
        for (destination, source) in [(lb, bounds.lower_b()), (ub, bounds.upper_b())] {
            copy_into(destination, source.as_slice(), source.iter().copied(), poll)?;
        }
        for (destination, source) in [(le, bounds.lower_a_err()), (ue, bounds.upper_a_err())] {
            if let Some(source) = source {
                copy_into(destination, source.as_slice(), source.iter().copied(), poll)?;
            }
        }
        poll.admit_elements(1, 2, DOMAIN_COMMIT)?;
        storage.lower_err_present[domain] = bounds.lower_a_err().is_some();
        storage.upper_err_present[domain] = bounds.upper_a_err().is_some();
    } else {
        coefficients(la, bounds.lower_a(), true, poll)?;
        coefficients(ua, bounds.upper_a(), false, poll)?;
        biases(lb, bounds.lower_b(), true, poll)?;
        biases(ub, bounds.upper_b(), false, poll)?;
        let lower = error_present(
            storage.lower_err_present[domain],
            bounds.lower_a_err(),
            la,
            poll,
        )?;
        let upper = error_present(
            storage.upper_err_present[domain],
            bounds.upper_a_err(),
            ua,
            poll,
        )?;
        if lower || upper {
            errors(
                le,
                storage.lower_err_present[domain],
                bounds.lower_a_err(),
                la,
                lower,
                poll,
            )?;
            errors(
                ue,
                storage.upper_err_present[domain],
                bounds.upper_a_err(),
                ua,
                upper,
                poll,
            )?;
            poll.admit_elements(1, 2, DOMAIN_COMMIT)?;
            storage.lower_err_present[domain] = true;
            storage.upper_err_present[domain] = true;
        }
    }
    poll.admit_elements(1, 4, DOMAIN_COMMIT)?;
    poll.checkpoint(DOMAIN_COMMIT)?;
    storage.active[domain] = true;
    if idx == pending.network_input_idx {
        pending.input_accumulated[domain] = true;
    }
    pending.charge.owner.check_live(poll)?;
    poll.checkpoint(PENDING_COMMIT)?;
    Ok(pending)
}

#[cfg(test)]
#[path = "bounded_merge_tests.rs"]
mod tests;
