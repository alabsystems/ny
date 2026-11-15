// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Necessary split constraints from completed, source-authenticated input rows.
//!
//! The result retains its producer borrow and original deadline. It is a
//! tile-local consumer, not a persistent RootGates sidecar or a new default
//! capability. The existing root-source authority is required by InputRowPair;
//! shape-valid or split-baked supplied gates cannot create that authority.

use std::time::Instant;

use ny_core::{NyError, Result};

use crate::certify::work::{LeaseClass, PersistentReservation, WorkRefusal, WorkRequest};

use super::clip::Halfspace;
use super::input_rows::InputRowPair;
use super::net::{check_deadline, TwinNet};
use super::root::RootGates;
use super::rounding::next_down;
use super::support_program::PreparedPrefix;

/// The original preactivation's branch premise, including zero on both sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputSplitSide {
    Inactive,
    Active,
}

/// A constraint whose original model/root/row owners must remain alive.
pub(super) struct ScopedInputHalfspace<'pair, 'source> {
    producer: &'pair InputRowPair<'source>,
    column: usize,
    side: InputSplitSide,
    halfspace: Halfspace,
    deadline: Instant,
    charged_error: f64,
    _resident: Option<PersistentReservation>,
}

impl ScopedInputHalfspace<'_, '_> {
    pub(super) fn halfspace(&self) -> &Halfspace {
        &self.halfspace
    }

    pub(super) fn original_target(&self) -> usize {
        self.producer
            .target(self.column)
            .expect("sealed original target")
    }

    pub(super) fn side(&self) -> InputSplitSide {
        self.side
    }

    pub(super) fn deadline(&self) -> Instant {
        self.deadline
    }

    pub(super) fn charged_error(&self) -> f64 {
        self.charged_error
    }
}

/// Local output-plus-resident census; not global memory admission or RSS.
pub(super) fn input_halfspace_bytes(resident: usize, inputs: usize) -> Option<usize> {
    resident
        .checked_add(size_of::<ScopedInputHalfspace<'_, '_>>())?
        .checked_add(inputs.checked_mul(size_of::<f64>())?)
}

/// Build one necessary condition without averaging independent row constants.
///
/// The caller already owns the whole-route admission; its resident census may
/// increase, but cannot understate this producer's visible live owners. The bounded dense output
/// is one input row, never a preactivation-by-input tableau. Coefficient errors
/// are paid over the original declared box, and final constant subtraction is
/// rounded downward separately from any later dual evaluation.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
pub(super) fn input_halfspace<'pair, 'source>(
    producer: &'pair InputRowPair<'source>,
    net: &TwinNet,
    root: &RootGates,
    relu_op: usize,
    targets: &[usize],
    column: usize,
    side: InputSplitSide,
    resident_bytes: usize,
    budget_bytes: usize,
    caller_deadline: Instant,
) -> Result<ScopedInputHalfspace<'pair, 'source>> {
    let deadline = caller_deadline.min(producer.deadline());
    check_deadline(deadline)?;
    ny_core::require_f64_interval_proof_environment()?;
    producer.validate_scope(net, root, relu_op, targets)?;
    let _row = match side {
        InputSplitSide::Inactive => producer.lower(column),
        InputSplitSide::Active => producer.upper(column),
    }
    .ok_or_else(|| NyError::InvalidSpec("input clipping: missing original target row".into()))?;
    let n_in = producer.n_in();
    let budget_bytes = budget_bytes.min(super::root_box::ROOT_SOURCE_BYTES_CAP);
    let resident_bytes = resident_bytes.max(producer.resident_floor(deadline)?);
    let required_bytes = input_halfspace_bytes(resident_bytes, n_in).unwrap_or(usize::MAX);
    if required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: "input clipping halfspace",
        });
    }
    let (halfspace, charged_error) = assemble_halfspace(
        producer,
        column,
        side,
        resident_bytes,
        budget_bytes,
        deadline,
        &|| check_deadline(deadline),
    )?;
    check_deadline(deadline)?;
    ny_core::require_f64_interval_proof_environment()?;
    Ok(ScopedInputHalfspace {
        producer,
        column,
        side,
        halfspace,
        deadline,
        charged_error,
        _resident: None,
    })
}

/// A necessary split condition from this exact original prepared prefix.
/// Both source and output use the producer's existing finite work ledger;
/// equal integer caps or a diagnostic root cannot authorize this path.
#[allow(clippy::too_many_arguments)]
pub(super) fn input_halfspace_prepared<'pair, 'source>(
    producer: &'pair InputRowPair<'source>,
    prefix: &PreparedPrefix,
    relu_op: usize,
    targets: &[usize],
    column: usize,
    side: InputSplitSide,
    caller_deadline: Instant,
) -> Result<ScopedInputHalfspace<'pair, 'source>> {
    let deadline = prefix
        .deadline(Some(caller_deadline.min(producer.deadline())))?
        .ok_or_else(|| NyError::InvalidSpec("input clipping: missing producer deadline".into()))?;
    producer.checkpoint(deadline)?;
    ny_core::require_f64_interval_proof_environment()?;
    producer.validate_prepared_prefix_binding(prefix, relu_op, targets.len(), deadline)?;
    if column >= producer.columns() {
        return Err(NyError::InvalidSpec(
            "input clipping: missing original target row".into(),
        ));
    }
    let root = prefix.root();
    let budget = root.budget();
    let resident_bytes = producer.resident_floor(deadline)?;
    let output_bytes = input_halfspace_bytes(0, producer.n_in())
        .and_then(|n| n.checked_add(256))
        .ok_or_else(|| NyError::InvalidSpec("input clipping: output capacity overflow".into()))?;
    let required_bytes = resident_bytes.checked_add(output_bytes).ok_or_else(|| {
        NyError::InvalidSpec("input clipping: total live capacity overflow".into())
    })?;
    let cap = super::root_box::ROOT_SOURCE_BYTES_CAP;
    if required_bytes > cap {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes: cap,
            site: "input clipping prepared total live",
        });
    }
    // Preserve the original N/output quote and additionally pay every ordered
    // target comparison before its first scan. The producer authenticates T in
    // constant time above; no temporary target vector or new byte claim is needed.
    let work = producer
        .n_in()
        .checked_mul(32)
        .and_then(|n| targets.len().checked_mul(32).and_then(|t| n.checked_add(t)))
        .and_then(|n| n.checked_add(1024))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| NyError::InvalidSpec("input clipping: work overflow".into()))?;
    let mut lease = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: output_bytes,
                work,
                ..WorkRequest::default()
            },
        )
        .map_err(prepared_refusal)?;
    let deadline = lease.deadline().map_or(deadline, |at| at.min(deadline));
    producer.validate_prepared_prefix_with_deadline(prefix, relu_op, targets, deadline)?;
    let (halfspace, charged_error) =
        assemble_halfspace(producer, column, side, 0, output_bytes, deadline, &|| {
            producer.checkpoint(deadline)
        })?;
    producer.checkpoint(deadline)?;
    ny_core::require_f64_interval_proof_environment()?;
    lease.mark_materialized().map_err(prepared_refusal)?;
    let resident = lease.commit_persistent().map_err(prepared_refusal)?;
    Ok(ScopedInputHalfspace {
        producer,
        column,
        side,
        halfspace,
        deadline,
        charged_error,
        _resident: Some(resident),
    })
}

/// Shared numerical assembly for the legacy and owned producer paths. Scope
/// authentication and admission happen at the respective checked entry.
fn assemble_halfspace(
    producer: &InputRowPair<'_>,
    column: usize,
    side: InputSplitSide,
    resident_bytes: usize,
    budget_bytes: usize,
    deadline: Instant,
    checkpoint: &dyn Fn() -> Result<()>,
) -> Result<(Halfspace, f64)> {
    checkpoint()?;
    let row = match side {
        InputSplitSide::Inactive => producer.lower(column),
        InputSplitSide::Active => producer.upper(column),
    }
    .ok_or_else(|| NyError::InvalidSpec("input clipping: missing original target row".into()))?;
    let charged_error = row.pointwise_error(deadline)?;
    checkpoint()?;
    let n_in = producer.n_in();
    let required_bytes = input_halfspace_bytes(resident_bytes, n_in).unwrap_or(usize::MAX);
    let mut a = Vec::new();
    a.try_reserve_exact(n_in)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: "input clipping halfspace allocation",
        })?;
    let actual_bytes = input_halfspace_bytes(resident_bytes, a.capacity()).unwrap_or(usize::MAX);
    if actual_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: actual_bytes,
            budget_bytes,
            site: "input clipping halfspace allocator capacity",
        });
    }
    for i in 0..n_in {
        if i.is_multiple_of(4096) {
            checkpoint()?;
        }
        a.push(0.0);
    }
    for (count, (index, coefficient, _)) in row.entries().enumerate() {
        if count.is_multiple_of(4096) {
            checkpoint()?;
        }
        a[index] = match side {
            InputSplitSide::Inactive => coefficient,
            InputSplitSide::Active => -coefficient,
        };
    }
    let nominal_bias = match side {
        InputSplitSide::Inactive => row.bias(),
        InputSplitSide::Active => -row.bias(),
    };
    let d = if charged_error == 0.0 {
        nominal_bias
    } else {
        next_down(nominal_bias - charged_error)
    };
    if !d.is_finite() {
        return Err(NyError::NumericalInstability(
            "input clipping: non-finite outward halfspace constant".into(),
        ));
    }
    checkpoint()?;
    Ok((Halfspace { a, d }, charged_error))
}

fn prepared_refusal(refusal: WorkRefusal) -> NyError {
    if refusal == WorkRefusal::Deadline {
        NyError::DeadlineExceeded("input clipping original deadline".into())
    } else {
        NyError::UnsupportedConfiguration(format!("input clipping original budget: {refusal}"))
    }
}

#[cfg(test)]
#[path = "input_clipping_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "input_clipping_prepared_tests.rs"]
mod prepared_tests;
