// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward GPU shader dispatch helpers (#3397).
//!
//! Contains shared types and compute pass submission/encoding helpers.
//! The batched backward path in `crown_backward.rs` builds bind groups
//! and encodes passes inline, using `encode_compute` to avoid per-dispatch
//! `queue.submit` overhead.

use super::super::WgpuDevice;
use super::crown_timestamps::CrownTimestampProfiler;
use ny_core::Result;

impl WgpuDevice {
    /// Encode a compute pass into an existing encoder without submitting.
    ///
    /// Used by the batched CROWN backward path to accumulate all dispatch
    /// passes into a single command encoder, eliminating per-dispatch submit
    /// overhead (#3397).
    pub(super) fn encode_compute(
        encoder: &mut wgpu::CommandEncoder,
        label: &'static str,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        dispatch: (u32, u32, u32),
        profiler: Option<&mut CrownTimestampProfiler>,
    ) -> Result<()> {
        let timestamp_writes = profiler
            .map(|profiler| profiler.allocate_pass(label))
            .transpose()?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
        Ok(())
    }
}
