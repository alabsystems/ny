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

    /// wgpu preserves subnormals and every sound DAG op is a certified enclosure
    /// (directed widening, NORMAL-range floors, Metal FTZ-safe by construction), so
    /// this backend advertises a verdict-legal sound DAG path — but ONLY on an adapter
    /// that passed the one-time IEEE-754 f32-model self-check (`ops/f32_selfcheck.rs`).
    /// An adapter with covert reduced precision / broken bitcast reports `false` here,
    /// so its verdicts fall back to the CPU sound graph loop (fail-safe). Cached.
    fn provides_sound_gpu_dag_ibp(&self) -> bool {
        self.verify_ieee_f32_model()
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
