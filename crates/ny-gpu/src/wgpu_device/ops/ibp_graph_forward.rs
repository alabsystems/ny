// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU-resident graph-DAG IBP forward pass (#4319).
//!
//! Implements [`GpuDagIbpForwardExt`] for [`WgpuDevice`], keeping lower/upper
//! bound buffers on GPU across all supported DAG ops (Linear, Conv2d, ReLU,
//! Add, View) and only reading back the final output bounds.

use ny_core::{GpuDagIbpForwardExt, GpuDagIbpModelPlan, GpuDagIbpPlanDesc, Result};

use super::super::WgpuDevice;

impl GpuDagIbpForwardExt for WgpuDevice {
    fn prepare_dag_model_plan(
        &self,
        plan: &GpuDagIbpPlanDesc,
    ) -> Result<Option<Box<dyn GpuDagIbpModelPlan>>> {
        if plan.ops.is_empty() {
            return Ok(None);
        }

        Ok(Some(Box::new(self.prepare_dag_model_plan_internal(plan)?)))
    }

    /// WGPU resident DAG IBP is temporarily excluded from verdict authority
    /// until overflow-sentinel taint is sticky across every supported DAG op.
    fn provides_sound_gpu_dag_ibp(&self) -> bool {
        false
    }

    fn prepare_sound_dag_model_plan(
        &self,
        plan: &GpuDagIbpPlanDesc,
    ) -> Result<Option<Box<dyn GpuDagIbpModelPlan>>> {
        if plan.ops.is_empty() {
            return Ok(None);
        }

        Ok(Some(Box::new(
            self.prepare_sound_dag_model_plan_internal(plan)?,
        )))
    }
}
