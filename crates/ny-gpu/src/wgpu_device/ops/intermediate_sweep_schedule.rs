// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pure, device-allocation-free planning for an Add/Sub-capable intermediate
//! sweep.
//!
//! Core validation proves the canonical reverse-DAG topology. This module adds
//! the device execution facts that core deliberately does not prescribe:
//!
//! * carrier rows retain their transaction-global identity for the whole walk;
//! * a future injection-only slot has no physical carrier until it is consumed;
//! * the first contribution to a slot moves ownership and later contributions
//!   merge into that allocation;
//! * Add/Sub owns one transient fork allocation, even when both edges converge
//!   on the same slot; and
//! * an incoming affine bias follows the lhs carrier only, so it is counted
//!   exactly once at the eventual convergence.
//!
//! The planner touches no WGPU object. It is therefore safe to run, including
//! all capacity and deadline checks, before the backend accepts a request.

use std::ops::Range;
use std::time::Instant;

use ny_core::{GpuBackwardOp, GpuBackwardSlot, GpuIntermediateSweepRequest, NyError, Result};

const DEADLINE_POLL_STRIDE: usize = 4096;

/// Exact location of one identity coefficient in the transaction-wide carrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DagInjectionReset {
    /// Immutable carrier row: `injection.row_offset + local_selected_row`.
    pub(super) carrier_row: usize,
    /// Flattened identity coordinate within this slot's dimension.
    pub(super) coordinate: usize,
}

/// Whether an incoming carrier becomes a slot's allocation or is merged into an
/// allocation already waiting at that slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DagContribution {
    Move,
    Merge,
}

/// Exact transform applied to the fork whose incoming bias is cleared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DagForkTransform {
    /// Forward `output = lhs + rhs`: copy coefficient centres unchanged.
    AddRhs,
    /// Forward `output = lhs - rhs`: negate lower and upper coefficient centres.
    /// Error radii and taint words are copied unchanged; lower/upper are not
    /// swapped because this is exact affine substitution, not interval negation.
    SubRhsNegateCenters,
}

/// Device action for one consumed output slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DagSlotAction {
    Unary {
        input: GpuBackwardSlot,
        destination: DagContribution,
    },
    Identity {
        input: GpuBackwardSlot,
        destination: DagContribution,
    },
    Fork {
        lhs: GpuBackwardSlot,
        rhs: GpuBackwardSlot,
        lhs_destination: DagContribution,
        /// This is computed after the lhs contribution. In particular, when
        /// `lhs == rhs`, the rhs necessarily merges into the lhs allocation.
        rhs_destination: DagContribution,
        rhs_transform: DagForkTransform,
    },
}

/// One canonical execution step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DagSlotStep {
    pub(super) op_index: usize,
    pub(super) output: GpuBackwardSlot,
    pub(super) output_dim: usize,
    /// True when this step must allocate an all-zero source carrier before
    /// applying `resets`; false means a propagated carrier is already waiting.
    pub(super) allocate_zero_source: bool,
    pub(super) resets: Vec<DagInjectionReset>,
    pub(super) action: DagSlotAction,
}

/// Carrier-only liveness at one step. These are exact logical buffer bytes for
/// the unfused typed-carrier lifecycle. Resident-fold scratch, retained caches,
/// uniforms, and final concretization staging are deliberately separate inputs
/// to [`DagCarrierPreflight::transaction_peak_bytes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DagCarrierUnitPreflight {
    pub(super) live_before_action_bytes: usize,
    pub(super) live_before_action_count: usize,
    /// Extra carrier allocated while the action still owns its source: the
    /// unary result carrier or the Add/Sub rhs fork. Identity uses zero.
    pub(super) transient_bytes: usize,
    pub(super) transient_count: usize,
    pub(super) live_after_action_bytes: usize,
    pub(super) live_after_action_count: usize,
}

/// Exact typed-carrier liveness for the planned physical execution units.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DagCarrierPreflight {
    pub(super) units: Vec<DagCarrierUnitPreflight>,
    pub(super) peak_carrier_bytes: usize,
    pub(super) peak_carrier_count: usize,
    pub(super) final_carrier_bytes: usize,
}

impl DagCarrierPreflight {
    /// Combine carrier liveness with exact step-local, non-carrier allocations.
    ///
    /// `unit_extra_bytes[i]` must count allocations live during execution unit
    /// `i` that
    /// are not represented by its source/result carrier (resident fold scratch,
    /// parameter buffers, queued staging, and pipeline-specific scratch). The
    /// final argument is the corresponding non-carrier working set while the
    /// final input carrier is concretized. Retained buffers remain live across
    /// both phases. All additions are checked; no saturating arithmetic is used
    /// in a memory receipt.
    pub(super) fn transaction_peak_bytes(
        &self,
        retained_bytes: usize,
        unit_extra_bytes: &[usize],
        final_concretize_extra_bytes: usize,
    ) -> Result<usize> {
        if unit_extra_bytes.len() != self.units.len() {
            return Err(invalid(format!(
                "DAG sweep unit scratch count {} != planned unit count {}",
                unit_extra_bytes.len(),
                self.units.len()
            )));
        }
        let mut peak = checked_add(
            retained_bytes,
            checked_add(
                self.final_carrier_bytes,
                final_concretize_extra_bytes,
                "final concretization working set",
            )?,
            "retained plus final concretization working set",
        )?;
        for (unit, &extra) in self.units.iter().zip(unit_extra_bytes) {
            let carrier_peak = checked_add(
                unit.live_before_action_bytes,
                unit.transient_bytes,
                "unit carrier peak",
            )?;
            peak = peak.max(checked_add(
                retained_bytes,
                checked_add(carrier_peak, extra, "unit working set")?,
                "retained plus unit working set",
            )?);
        }
        Ok(peak)
    }
}

/// A maximal sequence of unary steps that may use one resident fold call.
///
/// Every interior edge is a move into the immediately following output slot,
/// and no identity injection occurs at an interior slot. A merge, fan-out,
/// independent branch step, or injection ends the run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DagUnaryRun {
    pub(super) steps: Range<usize>,
}

/// One physical execution unit. A unary run retains only its first source and
/// final result carrier; its resident fold owns all intermediate ping-pong
/// scratch. This distinction is necessary for an exact peak when the first and
/// final dimensions are both wide but an interior dimension is narrow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum DagExecutionUnit {
    UnaryRun {
        steps: Range<usize>,
        output: GpuBackwardSlot,
        input: GpuBackwardSlot,
        destination: DagContribution,
    },
    Identity {
        step: usize,
    },
    Fork {
        step: usize,
    },
}

/// Complete pure schedule for one validated request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DagSweepSchedule {
    pub(super) rows: usize,
    pub(super) input_slot: GpuBackwardSlot,
    pub(super) steps: Vec<DagSlotStep>,
    pub(super) unary_runs: Vec<DagUnaryRun>,
    pub(super) execution: Vec<DagExecutionUnit>,
    pub(super) carriers: DagCarrierPreflight,
}

impl DagSweepSchedule {
    /// Validate and plan without allocating WGPU resources, waiting for a
    /// device lock, or dispatching. Unsupported unary kinds remain a clean
    /// pre-accept decline.
    pub(super) fn prepare(request: &GpuIntermediateSweepRequest<'_>) -> Result<Option<Self>> {
        request.validate()?;
        let plan = request.plan;
        let slot_count = plan.slot_dims.len();
        let rows = plan.total_rows;

        let mut resets_by_slot = vec![Vec::new(); slot_count];
        for (index, injection) in plan.injections.iter().enumerate() {
            poll_deadline(request.deadline, index, "while scheduling identity rows")?;
            let slot = injection.slot.index();
            let resets = &mut resets_by_slot[slot];
            resets
                .try_reserve(injection.selected_rows.len())
                .map_err(|error| {
                    invalid(format!(
                        "DAG sweep reset schedule allocation failed: {error}"
                    ))
                })?;
            for (local_row, &coordinate) in injection.selected_rows.iter().enumerate() {
                resets.push(DagInjectionReset {
                    carrier_row: injection
                        .row_offset
                        .checked_add(local_row)
                        .ok_or_else(|| invalid("DAG sweep carrier row overflow"))?,
                    coordinate: coordinate as usize,
                });
            }
        }

        let mut pending = vec![false; slot_count];
        let mut steps = Vec::new();
        steps
            .try_reserve_exact(plan.ops_backward.len())
            .map_err(|error| invalid(format!("DAG sweep step allocation failed: {error}")))?;

        for (op_index, op) in plan.ops_backward.iter().enumerate() {
            poll_deadline(request.deadline, op_index, "while planning DAG slots")?;
            if !supported_op(op) {
                return Ok(None);
            }
            let output = op_output(op);
            let output_index = output.index();
            let output_dim = plan.slot_dims[output_index];
            let allocate_zero_source = !pending[output_index];
            if allocate_zero_source {
                if resets_by_slot[output_index].is_empty() {
                    return Err(invalid(format!(
                        "DAG sweep slot {output_index} has neither a propagated carrier nor an injection"
                    )));
                }
                pending[output_index] = true;
            }

            let action = match op {
                GpuBackwardOp::Unary { input, .. } => {
                    let destination = contribution_for(*input, &pending);
                    pending[output_index] = false;
                    pending[input.index()] = true;
                    DagSlotAction::Unary {
                        input: *input,
                        destination,
                    }
                }
                GpuBackwardOp::Identity { input, .. } => {
                    let destination = contribution_for(*input, &pending);
                    pending[output_index] = false;
                    pending[input.index()] = true;
                    DagSlotAction::Identity {
                        input: *input,
                        destination,
                    }
                }
                GpuBackwardOp::Add { lhs, rhs, .. } | GpuBackwardOp::Sub { lhs, rhs, .. } => {
                    let lhs_destination = contribution_for(*lhs, &pending);
                    pending[output_index] = false;
                    pending[lhs.index()] = true;
                    // Deliberately sample after lhs: `x + x` and `x - x` use
                    // one destination carrier and merge the bias-cleared rhs.
                    let rhs_destination = contribution_for(*rhs, &pending);
                    pending[rhs.index()] = true;
                    let rhs_transform = if matches!(op, GpuBackwardOp::Sub { .. }) {
                        DagForkTransform::SubRhsNegateCenters
                    } else {
                        DagForkTransform::AddRhs
                    };
                    DagSlotAction::Fork {
                        lhs: *lhs,
                        rhs: *rhs,
                        lhs_destination,
                        rhs_destination,
                        rhs_transform,
                    }
                }
            };

            steps.push(DagSlotStep {
                op_index,
                output,
                output_dim,
                allocate_zero_source,
                resets: std::mem::take(&mut resets_by_slot[output_index]),
                action,
            });
        }

        deadline_check(request.deadline, "after planning DAG slots")?;
        let input_index = plan.input_slot.index();
        if pending
            .iter()
            .enumerate()
            .any(|(slot, &pending)| pending != (slot == input_index))
        {
            return Err(invalid(
                "DAG sweep logical carrier simulation did not end at input_slot only",
            ));
        }
        if resets_by_slot.iter().any(|resets| !resets.is_empty()) {
            return Err(invalid(
                "DAG sweep retained an identity injection after the final operation",
            ));
        }
        let (execution, unary_runs) = execution_units(&steps)?;
        let carriers = carrier_preflight(
            plan.slot_dims.as_ref(),
            plan.input_slot,
            rows,
            &steps,
            &execution,
            request.deadline,
        )?;
        Ok(Some(Self {
            rows,
            input_slot: plan.input_slot,
            steps,
            unary_runs,
            execution,
            carriers,
        }))
    }
}

fn supported_op(op: &GpuBackwardOp) -> bool {
    match op {
        GpuBackwardOp::Unary { layer, .. } => matches!(
            layer.as_ref(),
            ny_core::GpuCrownLayer::Linear { .. }
                | ny_core::GpuCrownLayer::Activation { .. }
                | ny_core::GpuCrownLayer::Conv2d { .. }
        ),
        GpuBackwardOp::Identity { .. } | GpuBackwardOp::Add { .. } | GpuBackwardOp::Sub { .. } => {
            true
        }
    }
}

fn op_output(op: &GpuBackwardOp) -> GpuBackwardSlot {
    match op {
        GpuBackwardOp::Unary { output, .. }
        | GpuBackwardOp::Identity { output, .. }
        | GpuBackwardOp::Add { output, .. }
        | GpuBackwardOp::Sub { output, .. } => *output,
    }
}

fn contribution_for(slot: GpuBackwardSlot, pending: &[bool]) -> DagContribution {
    if pending[slot.index()] {
        DagContribution::Merge
    } else {
        DagContribution::Move
    }
}

fn consume_output(
    output: GpuBackwardSlot,
    source_bytes: usize,
    pending: &mut [bool],
    live_bytes: &mut usize,
    live_count: &mut usize,
) -> Result<()> {
    let entry = pending
        .get_mut(output.index())
        .ok_or_else(|| invalid("DAG sweep output slot is out of range"))?;
    if !*entry {
        return Err(invalid("DAG sweep consumed an absent physical carrier"));
    }
    *entry = false;
    *live_bytes = live_bytes
        .checked_sub(source_bytes)
        .ok_or_else(|| invalid("DAG sweep live carrier bytes underflow"))?;
    *live_count = live_count
        .checked_sub(1)
        .ok_or_else(|| invalid("DAG sweep live carrier count underflow"))?;
    Ok(())
}

fn contribute(
    input: GpuBackwardSlot,
    carrier_bytes: usize,
    pending: &mut [bool],
    live_bytes: &mut usize,
    live_count: &mut usize,
) -> Result<()> {
    let entry = pending
        .get_mut(input.index())
        .ok_or_else(|| invalid("DAG sweep input slot is out of range"))?;
    if !*entry {
        *entry = true;
        *live_bytes = checked_add(*live_bytes, carrier_bytes, "live contributed carrier")?;
        *live_count = checked_add(*live_count, 1, "live contributed carrier count")?;
    }
    Ok(())
}

fn execution_units(steps: &[DagSlotStep]) -> Result<(Vec<DagExecutionUnit>, Vec<DagUnaryRun>)> {
    let mut units = Vec::new();
    let mut runs = Vec::new();
    let mut index = 0usize;
    while index < steps.len() {
        if !matches!(steps[index].action, DagSlotAction::Unary { .. }) {
            units.push(match steps[index].action {
                DagSlotAction::Identity { .. } => DagExecutionUnit::Identity { step: index },
                DagSlotAction::Fork { .. } => DagExecutionUnit::Fork { step: index },
                DagSlotAction::Unary { .. } => unreachable!("outer match excludes unary"),
            });
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        while index < steps.len() {
            let previous = &steps[index - 1];
            let next = &steps[index];
            let continues = matches!(
                previous.action,
                DagSlotAction::Unary {
                    input,
                    destination: DagContribution::Move,
                } if input == next.output
            ) && matches!(next.action, DagSlotAction::Unary { .. })
                && next.resets.is_empty()
                && !next.allocate_zero_source;
            if !continues {
                break;
            }
            index += 1;
        }
        let steps_range = start..index;
        let DagSlotAction::Unary { input, destination } = steps[index - 1].action else {
            return Err(invalid("DAG unary run does not end in a unary action"));
        };
        units.push(DagExecutionUnit::UnaryRun {
            steps: steps_range.clone(),
            output: steps[start].output,
            input,
            destination,
        });
        runs.push(DagUnaryRun { steps: steps_range });
    }
    Ok((units, runs))
}

fn carrier_preflight(
    slot_dims: &[usize],
    input_slot: GpuBackwardSlot,
    rows: usize,
    steps: &[DagSlotStep],
    execution: &[DagExecutionUnit],
    deadline: Instant,
) -> Result<DagCarrierPreflight> {
    let mut pending = vec![false; slot_dims.len()];
    let mut live_bytes = 0usize;
    let mut live_count = 0usize;
    let mut peak_bytes = 0usize;
    let mut peak_count = 0usize;
    let mut units = Vec::new();
    units
        .try_reserve_exact(execution.len())
        .map_err(|error| invalid(format!("DAG carrier preflight allocation failed: {error}")))?;

    for (unit_index, unit) in execution.iter().enumerate() {
        poll_deadline(deadline, unit_index, "while simulating carrier liveness")?;
        let step_index = match unit {
            DagExecutionUnit::UnaryRun { steps, .. } => steps.start,
            DagExecutionUnit::Identity { step } | DagExecutionUnit::Fork { step } => *step,
        };
        let step = steps
            .get(step_index)
            .ok_or_else(|| invalid("DAG execution unit references an absent step"))?;
        let output = step.output;
        let output_bytes = carrier_logical_bytes(rows, slot_dims[output.index()])?;
        let allocate_zero_source = !pending[output.index()];
        if allocate_zero_source != step.allocate_zero_source {
            return Err(invalid(format!(
                "DAG execution unit source allocation at slot {} disagrees with the slot schedule",
                output.index()
            )));
        }
        if allocate_zero_source {
            if step.resets.is_empty() {
                return Err(invalid(
                    "DAG execution unit needs a zero source but has no identity reset",
                ));
            }
            pending[output.index()] = true;
            live_bytes = checked_add(live_bytes, output_bytes, "live zero source carrier")?;
            live_count = checked_add(live_count, 1, "live zero source carrier count")?;
        }
        let before_bytes = live_bytes;
        let before_count = live_count;

        let (transient_bytes, transient_count) = match unit {
            DagExecutionUnit::UnaryRun {
                steps: run,
                output: run_output,
                input,
                destination,
            } => {
                if run.is_empty() || *run_output != output {
                    return Err(invalid("malformed DAG unary execution unit"));
                }
                let actual_destination = contribution_for(*input, &pending);
                if actual_destination != *destination {
                    return Err(invalid("DAG unary destination liveness drifted"));
                }
                let input_bytes = carrier_logical_bytes(rows, slot_dims[input.index()])?;
                consume_output(
                    output,
                    output_bytes,
                    &mut pending,
                    &mut live_bytes,
                    &mut live_count,
                )?;
                contribute(
                    *input,
                    input_bytes,
                    &mut pending,
                    &mut live_bytes,
                    &mut live_count,
                )?;
                (input_bytes, 1)
            }
            DagExecutionUnit::Identity { .. } => {
                let DagSlotAction::Identity { input, destination } = step.action else {
                    return Err(invalid(
                        "identity execution unit references a non-identity step",
                    ));
                };
                if contribution_for(input, &pending) != destination {
                    return Err(invalid("DAG identity destination liveness drifted"));
                }
                consume_output(
                    output,
                    output_bytes,
                    &mut pending,
                    &mut live_bytes,
                    &mut live_count,
                )?;
                contribute(
                    input,
                    output_bytes,
                    &mut pending,
                    &mut live_bytes,
                    &mut live_count,
                )?;
                (0, 0)
            }
            DagExecutionUnit::Fork { .. } => {
                let DagSlotAction::Fork {
                    lhs,
                    rhs,
                    lhs_destination,
                    rhs_destination,
                    ..
                } = step.action
                else {
                    return Err(invalid("fork execution unit references a non-fork step"));
                };
                if contribution_for(lhs, &pending) != lhs_destination {
                    return Err(invalid("DAG lhs destination liveness drifted"));
                }
                consume_output(
                    output,
                    output_bytes,
                    &mut pending,
                    &mut live_bytes,
                    &mut live_count,
                )?;
                contribute(
                    lhs,
                    output_bytes,
                    &mut pending,
                    &mut live_bytes,
                    &mut live_count,
                )?;
                if contribution_for(rhs, &pending) != rhs_destination {
                    return Err(invalid("DAG rhs destination liveness drifted"));
                }
                contribute(
                    rhs,
                    output_bytes,
                    &mut pending,
                    &mut live_bytes,
                    &mut live_count,
                )?;
                (output_bytes, 1)
            }
        };

        let unit_peak = checked_add(before_bytes, transient_bytes, "carrier unit peak")?;
        peak_bytes = peak_bytes.max(unit_peak).max(live_bytes);
        peak_count = peak_count
            .max(checked_add(
                before_count,
                transient_count,
                "carrier unit peak count",
            )?)
            .max(live_count);
        units.push(DagCarrierUnitPreflight {
            live_before_action_bytes: before_bytes,
            live_before_action_count: before_count,
            transient_bytes,
            transient_count,
            live_after_action_bytes: live_bytes,
            live_after_action_count: live_count,
        });
    }

    deadline_check(deadline, "after simulating carrier liveness")?;
    if pending
        .iter()
        .enumerate()
        .any(|(slot, &is_pending)| is_pending != (slot == input_slot.index()))
    {
        return Err(invalid(
            "DAG physical carrier liveness did not end at input_slot only",
        ));
    }
    let final_carrier_bytes = carrier_logical_bytes(rows, slot_dims[input_slot.index()])?;
    if live_bytes != final_carrier_bytes || live_count != 1 {
        return Err(invalid(format!(
            "DAG final carrier accounting ({live_bytes} bytes, {live_count} carriers) != \
             ({final_carrier_bytes} bytes, 1 carrier)"
        )));
    }
    Ok(DagCarrierPreflight {
        units,
        peak_carrier_bytes: peak_bytes,
        peak_carrier_count: peak_count,
        final_carrier_bytes,
    })
}

/// Logical bytes of one worded carrier with thirteen exact-sized buffers:
///
/// * four f32 coefficient/error lanes and four u32 coefficient/error words,
///   each `[rows, dim]`; and
/// * four f32 bias lanes plus one u32 row accumulator, each `[rows]`.
pub(super) fn carrier_logical_bytes(rows: usize, dim: usize) -> Result<usize> {
    if rows == 0 || dim == 0 {
        return Err(invalid("DAG sweep carrier rows and dim must be nonzero"));
    }
    let matrix = checked_mul(rows, dim, "carrier matrix elements")?;
    let matrix_bytes = checked_mul(matrix, 32, "eight carrier matrix lanes")?;
    let row_bytes = checked_mul(rows, 20, "five carrier row lanes")?;
    checked_add(matrix_bytes, row_bytes, "worded carrier bytes")
}

fn poll_deadline(deadline: Instant, index: usize, context: &str) -> Result<()> {
    if index.is_multiple_of(DEADLINE_POLL_STRIDE) {
        deadline_check(deadline, context)
    } else {
        Ok(())
    }
}

fn deadline_check(deadline: Instant, context: &str) -> Result<()> {
    if Instant::now() >= deadline {
        Err(NyError::DeadlineExceeded(format!(
            "WGPU DAG intermediate sweep deadline exceeded {context}"
        )))
    } else {
        Ok(())
    }
}

fn checked_add(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| invalid(format!("{label} byte/count overflow")))
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| invalid(format!("{label} byte/count overflow")))
}

fn invalid(message: impl Into<String>) -> NyError {
    NyError::InvalidSpec(message.into())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use ny_core::{
        CertifiedWeightError, GpuCrownLayer, GpuIntermediateInjection, GpuIntermediateSweepPlan,
    };

    use super::*;

    fn identity_plan(
        ops: impl Into<Arc<[GpuBackwardOp]>>,
        dims: impl Into<Arc<[usize]>>,
        injections: impl Into<Arc<[GpuIntermediateInjection]>>,
        rows: usize,
    ) -> GpuIntermediateSweepPlan {
        let slot_dims = dims.into();
        GpuIntermediateSweepPlan {
            graph_identity_sha256: [1; 32],
            bounds_identity_sha256: [2; 32],
            target_set_identity_sha256: [3; 32],
            ops_backward: ops.into(),
            input_slot: GpuBackwardSlot(u32::try_from(slot_dims.len() - 1).unwrap()),
            slot_dims,
            injections: injections.into(),
            total_rows: rows,
        }
    }

    fn injection(
        target_id: u64,
        slot: u32,
        rows: &[u32],
        row_offset: usize,
        dim: usize,
    ) -> GpuIntermediateInjection {
        GpuIntermediateInjection {
            target_id,
            slot: GpuBackwardSlot(slot),
            target_shape: Arc::from([dim]),
            selected_rows: Arc::from(rows),
            row_offset,
        }
    }

    fn request(plan: &GpuIntermediateSweepPlan) -> GpuIntermediateSweepRequest<'_> {
        let input_dim = plan.slot_dims[plan.input_slot.index()];
        let lower = vec![-1.0; input_dim].leak();
        let upper = vec![1.0; input_dim].leak();
        GpuIntermediateSweepRequest {
            plan,
            input_identity_sha256: [4; 32],
            input_lower: lower,
            input_upper: upper,
            deadline: Instant::now() + Duration::from_secs(10),
            max_device_bytes: 1 << 30,
        }
    }

    fn linear(dim: usize) -> GpuCrownLayer {
        let mut weight = vec![0.0; dim * dim];
        for index in 0..dim {
            weight[index * dim + index] = 1.0;
        }
        GpuCrownLayer::Linear {
            weight: Arc::from(weight),
            bias: None,
            out_features: dim,
            in_features: dim,
            cert_err: CertifiedWeightError::default(),
        }
    }

    #[test]
    fn diamond_forks_once_and_merges_at_convergence() {
        let plan = identity_plan(
            Arc::from([
                GpuBackwardOp::Add {
                    output: GpuBackwardSlot(0),
                    lhs: GpuBackwardSlot(1),
                    rhs: GpuBackwardSlot(2),
                },
                GpuBackwardOp::Identity {
                    output: GpuBackwardSlot(1),
                    input: GpuBackwardSlot(3),
                },
                GpuBackwardOp::Identity {
                    output: GpuBackwardSlot(2),
                    input: GpuBackwardSlot(3),
                },
            ]),
            Arc::from([2, 2, 2, 2]),
            Arc::from([injection(7, 0, &[1], 0, 2)]),
            1,
        );
        let schedule = DagSweepSchedule::prepare(&request(&plan)).unwrap().unwrap();
        let one = carrier_logical_bytes(1, 2).unwrap();
        assert_eq!(one, 84);
        assert_eq!(schedule.carriers.peak_carrier_bytes, 2 * one);
        assert_eq!(schedule.carriers.peak_carrier_count, 2);
        assert_eq!(schedule.carriers.final_carrier_bytes, one);
        assert!(matches!(
            schedule.steps[2].action,
            DagSlotAction::Identity {
                destination: DagContribution::Merge,
                ..
            }
        ));
    }

    #[test]
    fn sub_same_input_keeps_one_destination_and_negates_centres_only() {
        let plan = identity_plan(
            Arc::from([
                GpuBackwardOp::Sub {
                    output: GpuBackwardSlot(0),
                    lhs: GpuBackwardSlot(1),
                    rhs: GpuBackwardSlot(1),
                },
                GpuBackwardOp::Identity {
                    output: GpuBackwardSlot(1),
                    input: GpuBackwardSlot(2),
                },
            ]),
            Arc::from([3, 3, 3]),
            Arc::from([injection(9, 0, &[2], 0, 3)]),
            1,
        );
        let schedule = DagSweepSchedule::prepare(&request(&plan)).unwrap().unwrap();
        assert!(matches!(
            schedule.steps[0].action,
            DagSlotAction::Fork {
                lhs_destination: DagContribution::Move,
                rhs_destination: DagContribution::Merge,
                rhs_transform: DagForkTransform::SubRhsNegateCenters,
                ..
            }
        ));
        assert_eq!(schedule.carriers.units[0].live_after_action_count, 1);
        assert_eq!(schedule.carriers.peak_carrier_count, 2);
    }

    #[test]
    fn target_rows_never_compact_or_reorder() {
        let plan = identity_plan(
            Arc::from([
                GpuBackwardOp::Identity {
                    output: GpuBackwardSlot(0),
                    input: GpuBackwardSlot(1),
                },
                GpuBackwardOp::Identity {
                    output: GpuBackwardSlot(1),
                    input: GpuBackwardSlot(2),
                },
            ]),
            Arc::from([4, 4, 4]),
            Arc::from([
                injection(10, 0, &[1, 3], 0, 4),
                injection(20, 1, &[0, 2], 2, 4),
            ]),
            4,
        );
        let schedule = DagSweepSchedule::prepare(&request(&plan)).unwrap().unwrap();
        assert_eq!(
            schedule.steps[0].resets,
            vec![
                DagInjectionReset {
                    carrier_row: 0,
                    coordinate: 1,
                },
                DagInjectionReset {
                    carrier_row: 1,
                    coordinate: 3,
                },
            ]
        );
        assert_eq!(
            schedule.steps[1].resets,
            vec![
                DagInjectionReset {
                    carrier_row: 2,
                    coordinate: 0,
                },
                DagInjectionReset {
                    carrier_row: 3,
                    coordinate: 2,
                },
            ]
        );
        assert!(schedule.steps[0].allocate_zero_source);
        assert!(!schedule.steps[1].allocate_zero_source);
    }

    #[test]
    fn unary_runs_stop_at_an_injection_boundary() {
        let plan = identity_plan(
            Arc::from([
                GpuBackwardOp::Unary {
                    output: GpuBackwardSlot(0),
                    input: GpuBackwardSlot(1),
                    layer: Box::new(linear(2)),
                },
                GpuBackwardOp::Unary {
                    output: GpuBackwardSlot(1),
                    input: GpuBackwardSlot(2),
                    layer: Box::new(linear(2)),
                },
                GpuBackwardOp::Unary {
                    output: GpuBackwardSlot(2),
                    input: GpuBackwardSlot(3),
                    layer: Box::new(linear(2)),
                },
            ]),
            Arc::from([2, 2, 2, 2]),
            Arc::from([injection(10, 0, &[0], 0, 2), injection(20, 2, &[1], 1, 2)]),
            2,
        );
        let schedule = DagSweepSchedule::prepare(&request(&plan)).unwrap().unwrap();
        assert_eq!(
            schedule.unary_runs,
            vec![DagUnaryRun { steps: 0..2 }, DagUnaryRun { steps: 2..3 },]
        );
        assert_eq!(schedule.execution.len(), 2);
        assert_eq!(schedule.carriers.units.len(), 2);
    }

    #[test]
    fn fused_unary_run_does_not_count_internal_frontiers_as_live_carriers() {
        let plan = identity_plan(
            Arc::from([
                GpuBackwardOp::Unary {
                    output: GpuBackwardSlot(0),
                    input: GpuBackwardSlot(1),
                    layer: Box::new(GpuCrownLayer::Linear {
                        weight: Arc::from(vec![1.0; 2 * 7]),
                        bias: None,
                        out_features: 2,
                        in_features: 7,
                        cert_err: CertifiedWeightError::default(),
                    }),
                },
                GpuBackwardOp::Unary {
                    output: GpuBackwardSlot(1),
                    input: GpuBackwardSlot(2),
                    layer: Box::new(GpuCrownLayer::Linear {
                        weight: Arc::from(vec![1.0; 7 * 3]),
                        bias: None,
                        out_features: 7,
                        in_features: 3,
                        cert_err: CertifiedWeightError::default(),
                    }),
                },
            ]),
            Arc::from([2, 7, 3]),
            Arc::from([injection(10, 0, &[0], 0, 2)]),
            1,
        );
        let schedule = DagSweepSchedule::prepare(&request(&plan)).unwrap().unwrap();
        assert_eq!(schedule.execution.len(), 1);
        let expected = carrier_logical_bytes(1, 2).unwrap() + carrier_logical_bytes(1, 3).unwrap();
        assert_eq!(schedule.carriers.peak_carrier_bytes, expected);
        assert_eq!(schedule.carriers.peak_carrier_count, 2);
        assert!(
            schedule.carriers.peak_carrier_bytes < expected + carrier_logical_bytes(1, 7).unwrap(),
            "the internal seven-wide frontier belongs to resident-fold scratch, not a pending carrier"
        );
    }

    #[test]
    fn carrier_bytes_match_cifar_row_shapes_and_checked_peak() {
        assert_eq!(carrier_logical_bytes(512, 8192).unwrap(), 134_227_968);
        assert_eq!(carrier_logical_bytes(288, 8192).unwrap(), 75_503_232);
        assert_eq!(carrier_logical_bytes(288, 14_400).unwrap(), 132_716_160);

        let preflight = DagCarrierPreflight {
            units: vec![DagCarrierUnitPreflight {
                live_before_action_bytes: 100,
                live_before_action_count: 1,
                transient_bytes: 50,
                transient_count: 1,
                live_after_action_bytes: 75,
                live_after_action_count: 1,
            }],
            peak_carrier_bytes: 150,
            peak_carrier_count: 2,
            final_carrier_bytes: 75,
        };
        assert_eq!(
            preflight.transaction_peak_bytes(10, &[20], 30).unwrap(),
            180
        );
        assert!(preflight.transaction_peak_bytes(0, &[], 0).is_err());
    }

    #[test]
    fn expired_deadline_fails_before_a_schedule_is_returned() {
        let plan = identity_plan(
            Arc::from([GpuBackwardOp::Identity {
                output: GpuBackwardSlot(0),
                input: GpuBackwardSlot(1),
            }]),
            Arc::from([1, 1]),
            Arc::from([injection(1, 0, &[0], 0, 1)]),
            1,
        );
        let mut request = request(&plan);
        request.deadline = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap();
        assert!(matches!(
            DagSweepSchedule::prepare(&request),
            Err(NyError::DeadlineExceeded(_))
        ));
    }
}
