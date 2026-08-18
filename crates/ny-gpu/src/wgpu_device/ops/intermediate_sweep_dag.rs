// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Full-IEEE, word-preserving Add/Sub execution for `GpuIntermediateSweep`.
//!
//! Every pending frontier remains in an independently owned thirteen-buffer
//! carrier. Unary runs cross the existing authoritative resident fold through
//! its worded seed/keep seam; identity and binary edges use only encoder-ordered
//! device copies and the audited merge kernels. Each physical schedule unit is
//! fenced before the next allocation, making its scratch lifetime coincide
//! with the preflight unit whose bytes are charged below.

use std::mem::size_of;
use std::sync::Arc;

use bytemuck::Pod;
use ny_core::{
    GpuBackwardOp, GpuCrownLayer, GpuIntermediateSweepReceipt, GpuIntermediateSweepRequest,
    GpuIntermediateSweepResult, GpuIntermediateTargetResult, NyError, Result,
};

use super::super::shaders_intermediate_sweep::{
    SweepDagPipeline, SweepDagPipelines, SweepElementParams, SweepInjectParams, SweepInjectResetGpu,
};
use super::super::WgpuDevice;
use super::crown_backward_sound_resident::{
    resident_fold_plan, resident_fold_staging_capacity, ResidentCoeff, ResidentFoldPlan,
};
use super::intermediate_sweep::{
    capacity_decline, deadline_check, retained_device_bytes, SweepScope, SweepWorkReceipt,
    MAX_RESIDENT_CHAIN_LAYERS, MAX_RESIDENT_CHAIN_OPS, UNIFORM_UPLOAD_BYTES_PER_LAYER,
};
use super::intermediate_sweep_carrier::{
    DeviceSweepCarrier, SweepCarrierLayout, SweepDagAuthority,
};
use super::intermediate_sweep_schedule::{
    DagContribution, DagExecutionUnit, DagForkTransform, DagInjectionReset, DagSlotAction,
    DagSweepSchedule,
};

const ELEMENT_PARAM_BYTES: usize = size_of::<SweepElementParams>();
const INJECT_PARAM_BYTES: usize = size_of::<SweepInjectParams>();
/// A mapped-at-creation buffer with non-MAP usage may be implemented through
/// an internal transfer allocation. Charge the explicit buffer and one upload
/// twin, matching the hardened queue-write accounting rule.
const CONTROL_UPLOAD_RESIDENCY_FACTOR: usize = 2;

struct PreparedDag {
    schedule: DagSweepSchedule,
    /// One cloned layer slice for each physical unit; populated only for unary
    /// runs so execution cannot drift from the preflight grouping.
    unary_layers: Vec<Option<Vec<GpuCrownLayer>>>,
    unit_extra_bytes: Vec<usize>,
    retained_weight_ceiling: usize,
    final_extra_bytes: usize,
}

impl PreparedDag {
    fn peak_device_bytes(&self, retained: usize) -> Result<usize> {
        let persistent = checked_add(
            retained,
            self.retained_weight_ceiling,
            "retained buffers after DAG weight admission",
        )?;
        self.schedule.carriers.transaction_peak_bytes(
            persistent,
            &self.unit_extra_bytes,
            self.final_extra_bytes,
        )
    }
}

impl WgpuDevice {
    /// Cached full-IEEE DAG pipelines. Verdict qualification initializes this
    /// before authority is published, so an accepted sweep only reads it.
    pub(in crate::wgpu_device) fn intermediate_sweep_dag_pipelines(&self) -> &SweepDagPipelines {
        self.intermediate_sweep_dag_pipelines
            .get_or_init(|| SweepDagPipelines::create(&self.device, self.denorm_preserve_enabled()))
    }

    fn intermediate_sweep_dag_pipelines_cached(&self) -> Option<&SweepDagPipelines> {
        self.intermediate_sweep_dag_pipelines.get()
    }

    pub(super) fn run_intermediate_sweep_dag(
        &self,
        request: &GpuIntermediateSweepRequest<'_>,
    ) -> Result<Option<GpuIntermediateSweepResult>> {
        request.validate()?;
        if !request
            .plan
            .ops_backward
            .iter()
            .any(|op| matches!(op, GpuBackwardOp::Add { .. } | GpuBackwardOp::Sub { .. }))
        {
            return Ok(None);
        }
        if !self.provides_intermediate_sweep()
            || super::crown_backward_sound_resident::fold_coalesce_enabled()
            || !self.taint_words_armed()
        {
            return Ok(None);
        }
        let Some(_authority) = SweepDagAuthority::select(
            self.sound_gpu_authority_cached(),
            self.charged_flush_authority_cached().is_some(),
        ) else {
            // Charged-only adapters intentionally decline before pipeline
            // creation, allocation, reservation, or submission.
            return Ok(None);
        };
        let Some(schedule) = DagSweepSchedule::prepare(request)? else {
            return Ok(None);
        };
        // Full verdict qualification materializes resident, DAG, and sound
        // concretize pipelines. Keep every lookup before acceptance; a missing
        // cache is a typed refusal, never synchronous compilation inside the
        // checked transaction.
        let _ = self.resident_backward_pipelines();
        if self.device.limits().max_storage_buffers_per_shader_stage < 11 {
            return Ok(None);
        }
        let Some(pipelines) = self.intermediate_sweep_dag_pipelines_cached() else {
            return Err(NyError::UnsupportedOp(
                "WGPU DAG intermediate sweep pipelines were not materialized during qualification"
                    .into(),
            ));
        };
        if self.sound_concretize_pipeline_cached().is_none() {
            return Err(NyError::UnsupportedOp(
                "WGPU DAG intermediate sweep sound-concretize pipeline was not materialized during qualification"
                    .into(),
            ));
        }
        if !self.denorm_preserve_contract_intact() {
            return Err(NyError::UnsupportedOp(
                "WGPU DAG intermediate sweep pipeline loading contract was lost".into(),
            ));
        }
        let prepared = match self.prepare_dag_device_plan(request, schedule) {
            Ok(prepared) => prepared,
            Err(error) => return capacity_decline(error),
        };
        if let Err(error) = self.intermediate_sweep_concretize_preflight(
            request.plan.total_rows,
            request.plan.slot_dims[request.plan.input_slot.index()],
            request.input_lower,
            request.input_upper,
        ) {
            return capacity_decline(error);
        }
        deadline_check(request.deadline, "before DAG device preflight")?;

        let mut transaction =
            self.begin_gpu_checked_transaction("WGPU DAG intermediate sweep", request.deadline)?;
        self.require_full_ieee_dag_authority()?;

        let retained = retained_device_bytes(self)?;
        let peak_device_bytes = prepared.peak_device_bytes(retained)?;
        let Some(mut reservation) =
            self.reserve_intermediate_sweep_memory(request.max_device_bytes, peak_device_bytes)?
        else {
            return Ok(None);
        };
        deadline_check(request.deadline, "before accepting the DAG request")?;

        let _deadline_scope =
            crate::wgpu_device::CallLocalCrownDeadlineScope::arm(request.deadline);
        let work_scope = SweepScope::arm_dag()?;
        let coeff = self.execute_dag_carriers(request, &prepared, pipelines)?;
        self.require_full_ieee_dag_authority()?;
        let (lower, upper) = self.concretize_resident_coeff(
            &coeff,
            request.plan.total_rows,
            request.input_lower,
            request.input_upper,
        )?;
        deadline_check(request.deadline, "after DAG concretization")?;
        validate_publishable_bounds(&lower, &upper, request.plan.total_rows)?;
        let work = work_scope.finish()?;
        let targets = associate_targets(request, &lower, &upper)?;
        self.require_full_ieee_dag_authority()?;

        let receipt = make_receipt(request, peak_device_bytes, work);
        let validated =
            GpuIntermediateSweepResult::new_unvalidated(targets, receipt).validate(request)?;
        self.require_full_ieee_dag_authority()?;
        reservation.release()?;
        transaction.finish("WGPU DAG intermediate sweep")?;
        self.require_full_ieee_dag_authority()?;
        let (targets, receipt) = validated.into_parts();
        Ok(Some(GpuIntermediateSweepResult::new_unvalidated(
            targets, receipt,
        )))
    }

    fn require_full_ieee_dag_authority(&self) -> Result<()> {
        if self.sound_gpu_authority_cached() {
            Ok(())
        } else {
            Err(NyError::UnsupportedOp(
                "WGPU DAG intermediate sweep full-IEEE authority was lost; discarding the whole result"
                    .into(),
            ))
        }
    }

    fn prepare_dag_device_plan(
        &self,
        request: &GpuIntermediateSweepRequest<'_>,
        schedule: DagSweepSchedule,
    ) -> Result<PreparedDag> {
        if schedule.steps.len() > MAX_RESIDENT_CHAIN_OPS {
            return Err(capacity_error(
                schedule.steps.len(),
                MAX_RESIDENT_CHAIN_OPS,
                "DAG operations",
            ));
        }
        let unary_count = schedule
            .steps
            .iter()
            .filter(|step| matches!(step.action, DagSlotAction::Unary { .. }))
            .count();
        if unary_count > MAX_RESIDENT_CHAIN_LAYERS {
            return Err(capacity_error(
                unary_count,
                MAX_RESIDENT_CHAIN_LAYERS,
                "DAG unary layers",
            ));
        }

        let limits = self.device.limits();
        if limits.max_storage_buffers_per_shader_stage < 11 {
            return Err(capacity_error(
                11,
                limits.max_storage_buffers_per_shader_stage as usize,
                "storage buffers per shader stage",
            ));
        }
        for &dim in request.plan.slot_dims.iter() {
            let layout = SweepCarrierLayout::new(schedule.rows, dim)?.validate_device_limits(
                limits.max_buffer_size,
                limits.max_storage_buffer_binding_size,
            )?;
            checked_u32(layout.matrix_elements, "DAG carrier matrix elements")?;
        }

        let mut unary_layers = Vec::with_capacity(schedule.execution.len());
        let mut unit_extra_bytes = Vec::with_capacity(schedule.execution.len());
        let mut retained_weight_ceiling = 0usize;
        for unit in &schedule.execution {
            let step_index = unit_step_index(unit);
            let step = &schedule.steps[step_index];
            let mut extra =
                injection_allocation_bytes(schedule.rows, step.output_dim, &step.resets)?;
            let layers = match unit {
                DagExecutionUnit::UnaryRun {
                    steps, destination, ..
                } => {
                    let mut layers = Vec::with_capacity(steps.len());
                    for scheduled in &schedule.steps[steps.clone()] {
                        let op = request
                            .plan
                            .ops_backward
                            .get(scheduled.op_index)
                            .ok_or_else(|| invalid("DAG unary step references an absent op"))?;
                        let GpuBackwardOp::Unary { layer, .. } = op else {
                            return Err(invalid("DAG unary run references a non-unary op"));
                        };
                        layers.push((**layer).clone());
                    }
                    let fold = resident_fold_plan(
                        &layers,
                        schedule.rows,
                        schedule.rows,
                        step.output_dim,
                        limits.max_compute_workgroups_per_dimension,
                        limits.max_buffer_size,
                        limits.max_storage_buffer_binding_size,
                    )?;
                    let queued_fold_upload_bytes =
                        usize::try_from(resident_fold_staging_capacity(&layers, fold.n_domains)?)
                            .map_err(|_| invalid("DAG resident fold staging exceeds usize"))?;
                    let run_weights = layer_weight_bytes(&layers)?;
                    retained_weight_ceiling = checked_add(
                        retained_weight_ceiling,
                        run_weights,
                        "DAG retained weight ceiling",
                    )?;
                    extra = checked_add(
                        extra,
                        resident_unit_buffer_ceiling(
                            schedule.rows,
                            layers.len(),
                            fold,
                            queued_fold_upload_bytes,
                        )?,
                        "DAG unary resident working set",
                    )?;
                    // Queue uploads for newly materialized retained weights may
                    // coexist with the unit scratch until its fence.
                    extra = checked_add(extra, run_weights, "DAG retained-weight upload staging")?;
                    if *destination == DagContribution::Merge {
                        extra = checked_add(
                            extra,
                            control_upload_bytes(2 * ELEMENT_PARAM_BYTES)?,
                            "DAG unary merge params",
                        )?;
                    }
                    Some(layers)
                }
                DagExecutionUnit::Identity { .. } => {
                    let DagSlotAction::Identity { destination, .. } = step.action else {
                        return Err(invalid("identity unit references a non-identity action"));
                    };
                    if destination == DagContribution::Merge {
                        extra = checked_add(
                            extra,
                            control_upload_bytes(2 * ELEMENT_PARAM_BYTES)?,
                            "DAG identity merge params",
                        )?;
                    }
                    None
                }
                DagExecutionUnit::Fork { .. } => {
                    let DagSlotAction::Fork {
                        lhs_destination,
                        rhs_destination,
                        rhs_transform,
                        ..
                    } = step.action
                    else {
                        return Err(invalid("fork unit references a non-fork action"));
                    };
                    if rhs_transform == DagForkTransform::SubRhsNegateCenters {
                        extra = checked_add(
                            extra,
                            control_upload_bytes(ELEMENT_PARAM_BYTES)?,
                            "DAG Sub params",
                        )?;
                    }
                    for destination in [lhs_destination, rhs_destination] {
                        if destination == DagContribution::Merge {
                            extra = checked_add(
                                extra,
                                control_upload_bytes(2 * ELEMENT_PARAM_BYTES)?,
                                "DAG fork merge params",
                            )?;
                        }
                    }
                    None
                }
            };
            unary_layers.push(layers);
            unit_extra_bytes.push(extra);
        }

        let input_dim = request.plan.slot_dims[request.plan.input_slot.index()];
        let final_layout = SweepCarrierLayout::new(schedule.rows, input_dim)?;
        let download_staging = final_download_buffer_ceiling(final_layout)?;
        let concretize = concretize_buffer_ceiling(schedule.rows, input_dim)?;
        Ok(PreparedDag {
            schedule,
            unary_layers,
            unit_extra_bytes,
            retained_weight_ceiling,
            final_extra_bytes: download_staging.max(concretize),
        })
    }

    fn execute_dag_carriers(
        &self,
        request: &GpuIntermediateSweepRequest<'_>,
        prepared: &PreparedDag,
        pipelines: &SweepDagPipelines,
    ) -> Result<ResidentCoeff> {
        let schedule = &prepared.schedule;
        let mut pending: Vec<Option<DeviceSweepCarrier>> =
            (0..request.plan.slot_dims.len()).map(|_| None).collect();

        for (unit_index, unit) in schedule.execution.iter().enumerate() {
            deadline_check(request.deadline, "before a DAG execution unit")?;
            let step_index = unit_step_index(unit);
            let step = &schedule.steps[step_index];
            let output_index = step.output.index();
            if step.allocate_zero_source {
                if pending[output_index].is_some() {
                    return Err(invalid(
                        "DAG zero-source allocation would overwrite a carrier",
                    ));
                }
                let layout = SweepCarrierLayout::new(schedule.rows, step.output_dim)?;
                pending[output_index] = Some(DeviceSweepCarrier::allocate_zero_initialized(
                    &self.device,
                    layout,
                    "sweep_dag_source",
                )?);
            }
            let source = pending[output_index]
                .take()
                .ok_or_else(|| invalid("DAG execution consumed an absent carrier"))?;
            let mut unit_submitted = false;
            if !step.resets.is_empty() {
                self.submit_injection(&source, &step.resets, pipelines)?;
                unit_submitted = true;
            }

            match unit {
                DagExecutionUnit::UnaryRun {
                    input, destination, ..
                } => {
                    let layers = prepared.unary_layers[unit_index]
                        .as_deref()
                        .ok_or_else(|| invalid("DAG unary unit lost its prepared layers"))?;
                    let result =
                        self.crown_backward_sound_resident_sweep_carrier(layers, source)?;
                    unit_submitted = true;
                    if self.contribute_carrier(
                        &mut pending,
                        input.index(),
                        *destination,
                        result,
                        pipelines,
                    )? {
                        unit_submitted = true;
                    }
                }
                DagExecutionUnit::Identity { .. } => {
                    let DagSlotAction::Identity { input, destination } = step.action else {
                        return Err(invalid("identity unit action drifted"));
                    };
                    if self.contribute_carrier(
                        &mut pending,
                        input.index(),
                        destination,
                        source,
                        pipelines,
                    )? {
                        unit_submitted = true;
                    }
                }
                DagExecutionUnit::Fork { .. } => {
                    let DagSlotAction::Fork {
                        lhs,
                        rhs,
                        lhs_destination,
                        rhs_destination,
                        rhs_transform,
                    } = step.action
                    else {
                        return Err(invalid("fork unit action drifted"));
                    };
                    let rhs_carrier = DeviceSweepCarrier::allocate_zero_initialized(
                        &self.device,
                        source.layout,
                        "sweep_dag_rhs",
                    )?;
                    let mut encoder =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("sweep_dag_fork"),
                            });
                    source.encode_fork_biasless_rhs(&mut encoder, &rhs_carrier)?;
                    if rhs_transform == DagForkTransform::SubRhsNegateCenters {
                        self.encode_negate(&mut encoder, &rhs_carrier, pipelines)?;
                    }
                    encode_contribution(
                        self,
                        &mut encoder,
                        &mut pending,
                        lhs.index(),
                        lhs_destination,
                        source,
                        pipelines,
                    )?;
                    encode_contribution(
                        self,
                        &mut encoder,
                        &mut pending,
                        rhs.index(),
                        rhs_destination,
                        rhs_carrier,
                        pipelines,
                    )?;
                    self.submit_ticked(encoder.finish());
                    unit_submitted = true;
                }
            }

            if unit_submitted {
                self.wait_for_intermediate_sweep_unit("WGPU DAG execution unit")?;
            }
            self.require_full_ieee_dag_authority()?;
            deadline_check(request.deadline, "after a DAG execution unit")?;
        }

        let final_carrier = pending[request.plan.input_slot.index()]
            .take()
            .ok_or_else(|| invalid("DAG execution produced no input carrier"))?;
        if pending.iter().any(Option::is_some) {
            return Err(invalid("DAG execution retained a non-input carrier"));
        }
        self.download_dag_carrier(final_carrier, pipelines)
    }

    fn contribute_carrier(
        &self,
        pending: &mut [Option<DeviceSweepCarrier>],
        slot: usize,
        destination: DagContribution,
        carrier: DeviceSweepCarrier,
        pipelines: &SweepDagPipelines,
    ) -> Result<bool> {
        if destination == DagContribution::Move {
            let target = pending
                .get_mut(slot)
                .ok_or_else(|| invalid("DAG contribution slot is out of range"))?;
            if target.is_some() {
                return Err(invalid("DAG move contribution found an occupied slot"));
            }
            *target = Some(carrier);
            return Ok(false);
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sweep_dag_merge"),
            });
        encode_contribution(
            self,
            &mut encoder,
            pending,
            slot,
            destination,
            carrier,
            pipelines,
        )?;
        self.submit_ticked(encoder.finish());
        Ok(true)
    }

    fn submit_injection(
        &self,
        carrier: &DeviceSweepCarrier,
        resets: &[DagInjectionReset],
        pipelines: &SweepDagPipelines,
    ) -> Result<()> {
        let reset_gpu: Vec<SweepInjectResetGpu> = resets
            .iter()
            .map(|reset| {
                Ok(SweepInjectResetGpu {
                    carrier_row: checked_u32(reset.carrier_row, "DAG injection row")?,
                    coordinate: checked_u32(reset.coordinate, "DAG injection coordinate")?,
                })
            })
            .collect::<Result<_>>()?;
        let total = resets
            .len()
            .checked_mul(carrier.layout.dim)
            .ok_or_else(|| invalid("DAG injection element count overflow"))?;
        let (workgroups, stride) = self.dispatch_shape(total)?;
        let params = mapped_pod_buffer(
            self,
            "sweep_inject_params",
            wgpu::BufferUsages::UNIFORM,
            &SweepInjectParams {
                reset_count: checked_u32(resets.len(), "DAG injection count")?,
                rows: checked_u32(carrier.layout.rows, "DAG injection rows")?,
                dim: checked_u32(carrier.layout.dim, "DAG injection dim")?,
                total: checked_u32(total, "DAG injection total")?,
                stride,
                _padding: [0; 3],
            },
        );
        let reset_buf = mapped_slice_buffer(
            self,
            "sweep_inject_resets",
            wgpu::BufferUsages::STORAGE,
            &reset_gpu,
        )?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sweep_inject"),
            });
        encode_pass(
            self,
            &mut encoder,
            &pipelines.inject_values,
            &params,
            &[
                &reset_buf,
                &carrier.matrix.lower_center,
                &carrier.matrix.upper_center,
                &carrier.matrix.lower_radius,
                &carrier.matrix.upper_radius,
                &carrier.row.lower_bias,
                &carrier.row.upper_bias,
                &carrier.row.lower_bias_radius,
                &carrier.row.upper_bias_radius,
            ],
            workgroups,
        );
        encode_pass(
            self,
            &mut encoder,
            &pipelines.inject_words,
            &params,
            &[
                &reset_buf,
                &carrier.matrix.lower_center_word,
                &carrier.matrix.upper_center_word,
                &carrier.matrix.lower_radius_word,
                &carrier.matrix.upper_radius_word,
                &carrier.row.taint_rows,
            ],
            workgroups,
        );
        self.submit_ticked(encoder.finish());
        Ok(())
    }

    fn encode_negate(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        carrier: &DeviceSweepCarrier,
        pipelines: &SweepDagPipelines,
    ) -> Result<()> {
        let (workgroups, stride) = self.dispatch_shape(carrier.layout.matrix_elements)?;
        let params = mapped_pod_buffer(
            self,
            "sweep_negate_params",
            wgpu::BufferUsages::UNIFORM,
            &SweepElementParams {
                n: checked_u32(carrier.layout.matrix_elements, "DAG negate elements")?,
                stride,
                dim: checked_u32(carrier.layout.dim, "DAG negate dim")?,
                _padding: 0,
            },
        );
        encode_pass(
            self,
            encoder,
            &pipelines.negate_centers,
            &params,
            &[&carrier.matrix.lower_center, &carrier.matrix.upper_center],
            workgroups,
        );
        Ok(())
    }

    fn encode_merge(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        destination: &DeviceSweepCarrier,
        source: &DeviceSweepCarrier,
        pipelines: &SweepDagPipelines,
    ) -> Result<()> {
        if destination.layout != source.layout {
            return Err(NyError::shape_mismatch(
                vec![destination.layout.rows, destination.layout.dim],
                vec![source.layout.rows, source.layout.dim],
            ));
        }
        destination.validate_owned_sizes()?;
        source.validate_owned_sizes()?;
        let (matrix_workgroups, matrix_stride) =
            self.dispatch_shape(destination.layout.matrix_elements)?;
        let matrix_params = mapped_pod_buffer(
            self,
            "sweep_merge_matrix_params",
            wgpu::BufferUsages::UNIFORM,
            &SweepElementParams {
                n: checked_u32(destination.layout.matrix_elements, "DAG merge elements")?,
                stride: matrix_stride,
                dim: checked_u32(destination.layout.dim, "DAG merge dim")?,
                _padding: 0,
            },
        );
        for (
            dst_center,
            dst_radius,
            dst_center_word,
            dst_radius_word,
            src_center,
            src_radius,
            src_center_word,
            src_radius_word,
        ) in [
            (
                &destination.matrix.lower_center,
                &destination.matrix.lower_radius,
                &destination.matrix.lower_center_word,
                &destination.matrix.lower_radius_word,
                &source.matrix.lower_center,
                &source.matrix.lower_radius,
                &source.matrix.lower_center_word,
                &source.matrix.lower_radius_word,
            ),
            (
                &destination.matrix.upper_center,
                &destination.matrix.upper_radius,
                &destination.matrix.upper_center_word,
                &destination.matrix.upper_radius_word,
                &source.matrix.upper_center,
                &source.matrix.upper_radius,
                &source.matrix.upper_center_word,
                &source.matrix.upper_radius_word,
            ),
        ] {
            encode_pass(
                self,
                encoder,
                &pipelines.merge_matrix,
                &matrix_params,
                &[
                    dst_center,
                    dst_radius,
                    dst_center_word,
                    dst_radius_word,
                    src_center,
                    src_radius,
                    src_center_word,
                    src_radius_word,
                    &destination.row.taint_rows,
                ],
                matrix_workgroups,
            );
        }

        let (row_workgroups, row_stride) = self.dispatch_shape(destination.layout.rows)?;
        let row_params = mapped_pod_buffer(
            self,
            "sweep_merge_row_params",
            wgpu::BufferUsages::UNIFORM,
            &SweepElementParams {
                n: checked_u32(destination.layout.rows, "DAG merge rows")?,
                stride: row_stride,
                dim: 1,
                _padding: 0,
            },
        );
        encode_pass(
            self,
            encoder,
            &pipelines.merge_bias_rows,
            &row_params,
            &[
                &destination.row.lower_bias,
                &destination.row.lower_bias_radius,
                &destination.row.upper_bias,
                &destination.row.upper_bias_radius,
                &source.row.lower_bias,
                &source.row.lower_bias_radius,
                &source.row.upper_bias,
                &source.row.upper_bias_radius,
                &destination.row.taint_rows,
                &source.row.taint_rows,
            ],
            row_workgroups,
        );
        Ok(())
    }

    fn download_dag_carrier(
        &self,
        carrier: DeviceSweepCarrier,
        pipelines: &SweepDagPipelines,
    ) -> Result<ResidentCoeff> {
        carrier.validate_owned_sizes()?;
        let rows = carrier.layout.rows;
        let dim = carrier.layout.dim;
        let elements = carrier.layout.matrix_elements;
        self.run_gpu_checked_with_crown_deadline("worded DAG carrier download", || {
            let stage = |label: &str, bytes: u64| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            let st_lower_a = stage("sweep_dl_lower_a", carrier.layout.matrix_bytes);
            let st_upper_a = stage("sweep_dl_upper_a", carrier.layout.matrix_bytes);
            let st_lower_err = stage("sweep_dl_lower_err", carrier.layout.matrix_bytes);
            let st_upper_err = stage("sweep_dl_upper_err", carrier.layout.matrix_bytes);
            let st_lower_b = stage("sweep_dl_lower_b", carrier.layout.row_bytes);
            let st_upper_b = stage("sweep_dl_upper_b", carrier.layout.row_bytes);
            let st_lower_b_err = stage("sweep_dl_lower_b_err", carrier.layout.row_bytes);
            let st_upper_b_err = stage("sweep_dl_upper_b_err", carrier.layout.row_bytes);
            let st_rows = stage("sweep_dl_rows", carrier.layout.row_bytes);
            let (workgroups, stride) = self.dispatch_shape(elements)?;
            let params = mapped_pod_buffer(
                self,
                "sweep_finalize_rows_params",
                wgpu::BufferUsages::UNIFORM,
                &SweepElementParams {
                    n: checked_u32(elements, "DAG final word elements")?,
                    stride,
                    dim: checked_u32(dim, "DAG final word dim")?,
                    _padding: 0,
                },
            );
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sweep_dag_download"),
                });
            encode_pass(
                self,
                &mut encoder,
                &pipelines.finalize_rows,
                &params,
                &[
                    &carrier.matrix.lower_center_word,
                    &carrier.matrix.upper_center_word,
                    &carrier.matrix.lower_radius_word,
                    &carrier.matrix.upper_radius_word,
                    &carrier.row.taint_rows,
                ],
                workgroups,
            );
            for (source, destination) in [
                (&carrier.matrix.lower_center, &st_lower_a),
                (&carrier.matrix.upper_center, &st_upper_a),
                (&carrier.matrix.lower_radius, &st_lower_err),
                (&carrier.matrix.upper_radius, &st_upper_err),
            ] {
                encoder.copy_buffer_to_buffer(
                    source,
                    0,
                    destination,
                    0,
                    carrier.layout.matrix_bytes,
                );
            }
            for (source, destination) in [
                (&carrier.row.lower_bias, &st_lower_b),
                (&carrier.row.upper_bias, &st_upper_b),
                (&carrier.row.lower_bias_radius, &st_lower_b_err),
                (&carrier.row.upper_bias_radius, &st_upper_b_err),
                (&carrier.row.taint_rows, &st_rows),
            ] {
                encoder.copy_buffer_to_buffer(source, 0, destination, 0, carrier.layout.row_bytes);
            }
            self.submit_ticked(encoder.finish());
            let (mut values, taint_rows) = Self::read_sweep_carrier_batched(
                &self.device,
                &[
                    (&st_lower_a, elements),
                    (&st_upper_a, elements),
                    (&st_lower_err, elements),
                    (&st_upper_err, elements),
                    (&st_lower_b, rows),
                    (&st_upper_b, rows),
                    (&st_lower_b_err, rows),
                    (&st_upper_b_err, rows),
                ],
                (&st_rows, rows),
            )?;
            let upper_b_err = values.pop().expect("eight sweep value readbacks");
            let lower_b_err = values.pop().expect("eight sweep value readbacks");
            let upper_b = values.pop().expect("eight sweep value readbacks");
            let lower_b = values.pop().expect("eight sweep value readbacks");
            let upper_err = values.pop().expect("eight sweep value readbacks");
            let lower_err = values.pop().expect("eight sweep value readbacks");
            let upper_a = values.pop().expect("eight sweep value readbacks");
            let lower_a = values.pop().expect("eight sweep value readbacks");
            Ok(ResidentCoeff {
                lower_a,
                upper_a,
                lower_err,
                upper_err,
                lower_b,
                upper_b,
                lower_b_err,
                upper_b_err,
                dim,
                relu_grads: Vec::new(),
                beta_gather: Vec::new(),
                taint_rows: Some(taint_rows),
            })
        })
    }

    fn dispatch_shape(&self, elements: usize) -> Result<(u32, u32)> {
        let elements = checked_u32(elements, "DAG shader element count")?;
        let max_workgroups = self.device.limits().max_compute_workgroups_per_dimension;
        if max_workgroups == 0 {
            return Err(capacity_error(1, 0, "compute workgroups per dimension"));
        }
        let ideal = elements.div_ceil(256).max(1);
        let workgroups = ideal.min(max_workgroups);
        let stride = workgroups
            .checked_mul(256)
            .ok_or_else(|| invalid("DAG shader stride overflow"))?;
        Ok((workgroups, stride))
    }
}

fn encode_contribution(
    device: &WgpuDevice,
    encoder: &mut wgpu::CommandEncoder,
    pending: &mut [Option<DeviceSweepCarrier>],
    slot: usize,
    destination: DagContribution,
    carrier: DeviceSweepCarrier,
    pipelines: &SweepDagPipelines,
) -> Result<()> {
    let target = pending
        .get_mut(slot)
        .ok_or_else(|| invalid("DAG contribution slot is out of range"))?;
    match destination {
        DagContribution::Move => {
            if target.is_some() {
                return Err(invalid("DAG move contribution found an occupied slot"));
            }
            *target = Some(carrier);
        }
        DagContribution::Merge => {
            let target = target
                .as_ref()
                .ok_or_else(|| invalid("DAG merge contribution found an empty slot"))?;
            device.encode_merge(encoder, target, &carrier, pipelines)?;
        }
    }
    Ok(())
}

fn encode_pass(
    device: &WgpuDevice,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &SweepDagPipeline,
    params: &wgpu::Buffer,
    storage: &[&wgpu::Buffer],
    workgroups: u32,
) {
    super::intermediate_sweep::note_dispatches(1);
    let mut entries = Vec::with_capacity(storage.len() + 1);
    entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: params.as_entire_binding(),
    });
    for (index, buffer) in storage.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: (index + 1) as u32,
            resource: buffer.as_entire_binding(),
        });
    }
    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sweep_dag_bind_group"),
        layout: &pipeline.layout,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("sweep_dag_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline.pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups.max(1), 1, 1);
}

fn mapped_pod_buffer<T: Pod>(
    device: &WgpuDevice,
    label: &str,
    usage: wgpu::BufferUsages,
    value: &T,
) -> wgpu::Buffer {
    let bytes = bytemuck::bytes_of(value);
    let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage,
        mapped_at_creation: true,
    });
    buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(bytes);
    buffer.unmap();
    super::intermediate_sweep::note_host_to_device(bytes.len());
    buffer
}

fn mapped_slice_buffer<T: Pod>(
    device: &WgpuDevice,
    label: &str,
    usage: wgpu::BufferUsages,
    values: &[T],
) -> Result<wgpu::Buffer> {
    let bytes = bytemuck::cast_slice(values);
    if bytes.is_empty() {
        return Err(invalid("DAG mapped slice buffer cannot be empty"));
    }
    let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage,
        mapped_at_creation: true,
    });
    buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(bytes);
    buffer.unmap();
    super::intermediate_sweep::note_host_to_device(bytes.len());
    Ok(buffer)
}

fn unit_step_index(unit: &DagExecutionUnit) -> usize {
    match unit {
        DagExecutionUnit::UnaryRun { steps, .. } => steps.start,
        DagExecutionUnit::Identity { step } | DagExecutionUnit::Fork { step } => *step,
    }
}

fn injection_allocation_bytes(
    rows: usize,
    dim: usize,
    resets: &[DagInjectionReset],
) -> Result<usize> {
    if resets.is_empty() {
        return Ok(0);
    }
    checked_u32(rows, "DAG injection rows")?;
    checked_u32(dim, "DAG injection dim")?;
    checked_u32(resets.len(), "DAG injection count")?;
    checked_u32(
        resets
            .len()
            .checked_mul(dim)
            .ok_or_else(|| invalid("DAG injection total overflow"))?,
        "DAG injection total",
    )?;
    control_upload_bytes(checked_add(
        checked_mul(
            resets.len(),
            size_of::<SweepInjectResetGpu>(),
            "DAG injection reset upload",
        )?,
        INJECT_PARAM_BYTES,
        "DAG injection allocations",
    )?)
}

fn control_upload_bytes(explicit_bytes: usize) -> Result<usize> {
    checked_mul(
        explicit_bytes,
        CONTROL_UPLOAD_RESIDENCY_FACTOR,
        "DAG explicit control buffer plus upload twin",
    )
}

fn resident_unit_buffer_ceiling(
    rows: usize,
    layer_count: usize,
    fold: ResidentFoldPlan,
    queued_fold_upload_bytes: usize,
) -> Result<usize> {
    // Same audited categories as the hardened chain estimator, minus the final
    // host staging (keep mode has none) and retained weights (charged once in
    // PreparedDag). Doubling charges all explicit resident buffers again as a
    // conservative ceiling for queue-owned upload allocations.
    let elements = checked_add(
        checked_mul(fold.a_elems, 32, "DAG resident coefficient workspaces")?,
        checked_add(
            checked_mul(fold.max_gemm_out, 6, "DAG resident GEMM workspaces")?,
            checked_add(
                checked_mul(fold.slope_dim, 7, "DAG resident activation workspaces")?,
                checked_add(
                    checked_mul(rows, 16, "DAG resident row workspaces")?,
                    checked_mul(fold.max_dim, 4, "DAG resident vector workspaces")?,
                    "DAG resident row/vector workspaces",
                )?,
                "DAG resident activation workspaces",
            )?,
            "DAG resident GEMM workspaces",
        )?,
        "DAG resident workspaces",
    )?;
    let explicit = checked_add(
        checked_mul(elements, size_of::<f32>(), "DAG resident buffer bytes")?,
        checked_mul(
            layer_count,
            UNIFORM_UPLOAD_BYTES_PER_LAYER,
            "DAG resident uniform/upload allowance",
        )?,
        "DAG resident explicit buffers",
    )?;
    checked_add(
        checked_mul(
            explicit,
            2,
            "DAG resident buffers plus initial queued uploads",
        )?,
        queued_fold_upload_bytes,
        "DAG resident buffers plus every cumulative fold upload",
    )
}

fn layer_weight_bytes(layers: &[GpuCrownLayer]) -> Result<usize> {
    let elements = layers.iter().try_fold(0usize, |total, layer| {
        let count = match layer {
            GpuCrownLayer::Linear { weight, .. } => weight.len(),
            GpuCrownLayer::Conv2d { weight_col, .. } => weight_col.len(),
            _ => 0,
        };
        total.checked_add(count)
    });
    let elements = elements.ok_or_else(|| invalid("DAG retained weight count overflow"))?;
    checked_mul(
        elements,
        3 * size_of::<f32>(),
        "DAG retained raw/abs/transpose weights",
    )
}

fn concretize_buffer_ceiling(rows: usize, dim: usize) -> Result<usize> {
    let coeff = checked_mul(rows, dim, "DAG concretize coefficients")?;
    let elements = checked_add(
        checked_mul(coeff, 4, "DAG concretize coefficient streams")?,
        checked_add(
            checked_mul(dim, 2, "DAG concretize box")?,
            checked_mul(rows, 6, "DAG concretize row streams")?,
            "DAG concretize box/row streams",
        )?,
        "DAG concretize elements",
    )?;
    let explicit = checked_add(
        checked_mul(elements, size_of::<f32>(), "DAG concretize bytes")?,
        4096,
        "DAG concretize uniforms",
    )?;
    checked_mul(explicit, 2, "DAG concretize buffers plus queued uploads")
}

fn final_download_buffer_ceiling(layout: SweepCarrierLayout) -> Result<usize> {
    checked_add(
        checked_add(
            checked_mul(
                usize::try_from(layout.matrix_bytes)
                    .map_err(|_| invalid("DAG matrix bytes do not fit usize"))?,
                4,
                "DAG final f32 matrix staging",
            )?,
            checked_mul(
                usize::try_from(layout.row_bytes)
                    .map_err(|_| invalid("DAG row bytes do not fit usize"))?,
                5,
                "DAG final row staging",
            )?,
            "DAG final download staging",
        )?,
        control_upload_bytes(ELEMENT_PARAM_BYTES)?,
        "DAG final row-fold params",
    )
}

fn validate_publishable_bounds(lower: &[f32], upper: &[f32], rows: usize) -> Result<()> {
    if lower.len() != rows || upper.len() != rows {
        return Err(NyError::InternalError(
            "WGPU DAG sweep returned a partial carrier".into(),
        ));
    }
    if lower
        .iter()
        .zip(upper)
        .any(|(&lo, &hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
    {
        return Err(NyError::InternalError(
            "WGPU DAG sweep returned a non-publishable interval".into(),
        ));
    }
    Ok(())
}

fn associate_targets(
    request: &GpuIntermediateSweepRequest<'_>,
    lower: &[f32],
    upper: &[f32],
) -> Result<Vec<GpuIntermediateTargetResult>> {
    let mut targets = Vec::with_capacity(request.plan.injections.len());
    for injection in request.plan.injections.iter() {
        deadline_check(request.deadline, "while associating DAG results")?;
        let end = injection
            .row_offset
            .checked_add(injection.selected_rows.len())
            .ok_or_else(|| invalid("DAG result slice overflow"))?;
        targets.push(GpuIntermediateTargetResult {
            target_id: injection.target_id,
            row_offset: injection.row_offset,
            selected_rows: Arc::clone(&injection.selected_rows),
            lower_bounds: lower
                .get(injection.row_offset..end)
                .ok_or_else(|| invalid("DAG lower result slice missing"))?
                .to_vec(),
            upper_bounds: upper
                .get(injection.row_offset..end)
                .ok_or_else(|| invalid("DAG upper result slice missing"))?
                .to_vec(),
        });
    }
    Ok(targets)
}

fn make_receipt(
    request: &GpuIntermediateSweepRequest<'_>,
    peak_device_bytes: usize,
    work: SweepWorkReceipt,
) -> GpuIntermediateSweepReceipt {
    GpuIntermediateSweepReceipt {
        graph_identity_sha256: request.plan.graph_identity_sha256,
        input_identity_sha256: request.input_identity_sha256,
        bounds_identity_sha256: request.plan.bounds_identity_sha256,
        target_set_identity_sha256: request.plan.target_set_identity_sha256,
        requested_targets: request.plan.injections.len(),
        completed_targets: request.plan.injections.len(),
        requested_rows: request.plan.total_rows,
        completed_rows: request.plan.total_rows,
        peak_device_bytes,
        dispatches: work.dispatches,
        host_to_device_bytes: work.host_to_device_bytes,
        device_to_host_bytes: work.device_to_host_bytes,
        readbacks: work.readbacks,
        submits: work.submits,
        synchronizations: work.synchronizations,
        waves: 1,
    }
}

fn capacity_error(requested: usize, capacity: usize, unit: &'static str) -> NyError {
    NyError::GpuBatchCapacityExceeded {
        requested,
        capacity,
        unit,
        site: "WGPU intermediate DAG preflight",
    }
}

fn checked_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("{label} exceeds u32")))
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
    use super::*;

    #[test]
    fn every_mapped_control_buffer_charges_an_upload_twin() {
        let resets = [
            DagInjectionReset {
                carrier_row: 0,
                coordinate: 1,
            },
            DagInjectionReset {
                carrier_row: 7,
                coordinate: 3,
            },
        ];
        // Two 8-byte reset records + one 32-byte uniform, each charged twice.
        assert_eq!(injection_allocation_bytes(8, 4, &resets).unwrap(), 96);
        assert_eq!(control_upload_bytes(ELEMENT_PARAM_BYTES).unwrap(), 32);
        assert_eq!(control_upload_bytes(2 * ELEMENT_PARAM_BYTES).unwrap(), 64);
    }

    #[test]
    fn final_download_counts_four_matrix_and_five_row_staging_buffers() {
        let layout = SweepCarrierLayout::new(288, 8192).unwrap();
        assert_eq!(final_download_buffer_ceiling(layout).unwrap(), 37_754_528);
    }

    #[test]
    fn resident_unit_adds_the_complete_cumulative_fold_staging() {
        let fold = ResidentFoldPlan {
            num_specs_u32: 1,
            num_specs_per_dom_u32: 1,
            n_domains: 1,
            seed_elems: 1,
            final_dim: 1,
            max_dim: 1,
            max_gemm_out: 1,
            a_elems: 1,
            slope_dim: 1,
            max_wg: 1,
        };
        let without = resident_unit_buffer_ceiling(1, 1, fold, 0).unwrap();
        let queued = 7 * 1024 * 1024;
        let with = resident_unit_buffer_ceiling(1, 1, fold, queued).unwrap();
        assert_eq!(with - without, queued);
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_verdict_device};
    use ny_core::{
        CertifiedWeightError, GpuBackwardSlot, GpuCrownBackward, GpuIntermediateInjection,
        GpuIntermediateSweepPlan,
    };
    use ny_test_utils::env::ScopedEnvVar;
    use std::time::{Duration, Instant};

    fn certified_zero_linear(bias_abs_err: f32) -> GpuCrownLayer {
        GpuCrownLayer::Linear {
            weight: Arc::from([0.0]),
            bias: None,
            out_features: 1,
            in_features: 1,
            cert_err: CertifiedWeightError {
                weight_rel_err: 0.0,
                bias_abs_err,
            },
        }
    }

    fn certified_residual_plan(subtract: bool) -> GpuIntermediateSweepPlan {
        let residual = if subtract {
            GpuBackwardOp::Sub {
                output: GpuBackwardSlot(0),
                lhs: GpuBackwardSlot(1),
                rhs: GpuBackwardSlot(2),
            }
        } else {
            GpuBackwardOp::Add {
                output: GpuBackwardSlot(0),
                lhs: GpuBackwardSlot(1),
                rhs: GpuBackwardSlot(2),
            }
        };
        GpuIntermediateSweepPlan {
            graph_identity_sha256: [61 + u8::from(subtract); 32],
            bounds_identity_sha256: [63 + u8::from(subtract); 32],
            target_set_identity_sha256: [65 + u8::from(subtract); 32],
            ops_backward: Arc::from([
                residual,
                GpuBackwardOp::Unary {
                    output: GpuBackwardSlot(1),
                    input: GpuBackwardSlot(3),
                    layer: Box::new(certified_zero_linear(0.25)),
                },
                GpuBackwardOp::Unary {
                    output: GpuBackwardSlot(2),
                    input: GpuBackwardSlot(3),
                    layer: Box::new(certified_zero_linear(0.5)),
                },
            ]),
            slot_dims: Arc::from([1, 1, 1, 1]),
            input_slot: GpuBackwardSlot(3),
            injections: Arc::from([GpuIntermediateInjection {
                target_id: 601 + u64::from(subtract),
                slot: GpuBackwardSlot(0),
                target_shape: Arc::from([1]),
                selected_rows: Arc::from([0]),
                row_offset: 0,
            }]),
            total_rows: 1,
        }
    }

    #[test]
    fn live_add_and_sub_merge_both_certified_bias_radii() {
        let _serial = gpu_test_serial_guard();
        let _coalesce = ScopedEnvVar::unset("NY_FOLD_COALESCE");
        let _eft = ScopedEnvVar::unset("NY_EFT_ERR");
        let _words = ScopedEnvVar::unset("NY_GPU_TAINT_WORDS");
        let device = require_verdict_device();
        assert!(device.provides_sound_intermediate_sweep());

        // Each fork arm starts with coefficient magnitude one. The documented
        // certified-bias charge is d * (sum(|a|) + sum(err)), so these exact
        // seeds contribute radii 0.25 and 0.5. Minkowski addition applies to
        // both a+b and a-b: negating the Sub RHS centre must not negate its
        // radius. The exact intended residual radius is therefore 0.75.
        let expected_radius = 0.25f64 + 0.5f64;
        let input_lower = [0.0];
        let input_upper = [0.0];
        for subtract in [false, true] {
            let plan = certified_residual_plan(subtract);
            let request = GpuIntermediateSweepRequest {
                plan: &plan,
                input_identity_sha256: [67 + u8::from(subtract); 32],
                input_lower: &input_lower,
                input_upper: &input_upper,
                deadline: Instant::now() + Duration::from_secs(30),
                max_device_bytes: 256 << 20,
            };
            let result = device
                .crown_backward_gpu_sound_intermediate_sweep(&request)
                .expect("live certified residual sweep")
                .expect("qualified residual DAG must be accepted")
                .validate(&request)
                .expect("certified residual atomic result validation");
            let target = &result.targets()[0];
            let lower = f64::from(target.lower_bounds[0]);
            let upper = f64::from(target.upper_bounds[0]);
            assert!(lower.is_finite() && upper.is_finite());
            assert!(
                lower <= -expected_radius,
                "{} dropped certified lower-radius evidence: {lower:e}",
                if subtract { "Sub" } else { "Add" }
            );
            assert!(
                upper >= expected_radius,
                "{} dropped certified upper-radius evidence: {upper:e}",
                if subtract { "Sub" } else { "Add" }
            );
            assert!(
                lower > -0.751 && upper < 0.751,
                "{} returned a fallback/sentinel interval [{lower:e}, {upper:e}]",
                if subtract { "Sub" } else { "Add" }
            );
            let receipt = result.receipt();
            assert_eq!((receipt.completed_targets, receipt.completed_rows), (1, 1));
            assert!(receipt.host_to_device_bytes > 0);
            assert!(receipt.dispatches > 0 && receipt.submits > 0);
            assert!(receipt.readbacks > 0 && receipt.device_to_host_bytes > 0);
        }
    }

    #[test]
    fn live_dag_post_submit_validation_error_discards_result_and_poisons_private_device() {
        let _serial = gpu_test_serial_guard();
        // Deliberately use a private device: poisoning is permanent by design,
        // so this fault discriminator must not contaminate the shared verdict
        // device used by the rest of the live suite.
        let device = WgpuDevice::new().expect("dedicated WGPU fault-test device");
        let outcome: Result<Option<()>> = (|| {
            let mut transaction = device.begin_gpu_checked_transaction(
                "scripted DAG post-submit validation fault",
                Instant::now() + Duration::from_secs(10),
            )?;
            let _reservation = device
                .reserve_intermediate_sweep_memory(4096, 4096)?
                .ok_or_else(|| {
                    NyError::InternalError("scripted DAG reservation declined".into())
                })?;
            let _scope = SweepScope::arm_dag()?;

            let encoder = device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("scripted DAG pre-fault submit"),
                });
            device.submit_ticked(encoder.finish());

            // This is the same deterministic validation fault used by the
            // checked-transaction nesting discriminator. It occurs only after
            // the DAG scope has recorded a real queue submission.
            let invalid_size = device.device.limits().max_buffer_size.saturating_add(1);
            let _invalid = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("intentional-invalid-buffer-after-DAG-submit"),
                size: invalid_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            transaction.finish("scripted DAG post-submit validation fault")?;

            // Publication is reachable only after every checked scope passes.
            // This value is the discriminator: the validation error must make
            // the closure return Err rather than exposing even an empty result.
            Ok(Some(()))
        })();

        let error = outcome.expect_err("post-submit validation fault must discard the result");
        assert!(
            error.to_string().contains("validation"),
            "transaction-owned validation scope returned the wrong error: {error}"
        );
        assert_eq!(
            *device.intermediate_sweep_reserved_bytes.lock().unwrap(),
            usize::MAX,
            "an undrained accepted DAG submission must permanently poison its device ledger"
        );
        assert!(
            device
                .reserve_intermediate_sweep_memory(4096, 1)
                .expect("poisoned-ledger query")
                .is_none(),
            "a poisoned device must decline every later sweep reservation"
        );
    }
}
