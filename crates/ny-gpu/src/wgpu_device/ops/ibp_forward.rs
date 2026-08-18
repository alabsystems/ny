// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU-resident IBP forward pass (#4081).
//!
//! Keeps lower/upper bound buffers on GPU across all supported layers in a
//! resident chain (Linear / Conv2d / ReLU / View) and only reads back the
//! final output bounds. The cached-plan details live in `ibp_forward_plan.rs`
//! so this entry point stays under the repo's file-size ceiling.

use ny_core::{
    GpuIbpForward, GpuIbpForwardExt, GpuIbpLayer, GpuIbpModelPlan, GpuIbpResult, Result,
};

use super::super::WgpuDevice;

pub(super) fn create_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    // #gpu-pool-highwater: this helper is the OTHER allocation choke point --
    // the IBP/DAG plan builders allocate per-op lower/upper buffers through it,
    // entirely outside `BufferPool`, so a ledger that only watched the pool
    // would attribute nothing to them. These are owned by their plan and freed
    // when it drops; the plan caches are cleared via `clear_crown_working_set`.
    crate::gpu_memory_ledger::record_alloc(label, size);
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

impl GpuIbpForward for WgpuDevice {
    fn ibp_forward_gpu(
        &self,
        layers: &[GpuIbpLayer],
        input_lower: &[f32],
        input_upper: &[f32],
        input_shape: &[usize],
    ) -> Result<GpuIbpResult> {
        if layers.is_empty() {
            return Ok(GpuIbpResult {
                lower_bounds: input_lower.to_vec(),
                upper_bounds: input_upper.to_vec(),
                output_shape: input_shape.to_vec(),
            });
        }

        self.prepare_model_plan_internal(layers, input_shape)?
            .ibp_forward_cached(input_lower, input_upper, input_shape)
    }

    /// SOUND (verdict-legal) GPU IBP forward (`docs/SOUND_GPU_IBP_PLAN.md` §3.1
    /// keystone + §6.3). Dispatches the sound resident driver for Linear/ReLU dense
    /// chains; any other layer kind returns `Err(UnsupportedOp)` so the caller takes
    /// the proven-sound CPU fallback (verdict-safe). A wgpu error inside the driver
    /// is likewise turned into `Err` (never a value from a failed op).
    fn ibp_forward_gpu_sound(
        &self,
        layers: &[GpuIbpLayer],
        input_lower: &[f32],
        input_upper: &[f32],
        input_shape: &[usize],
    ) -> Result<GpuIbpResult> {
        self.ibp_forward_gpu_sound_dispatch(layers, input_lower, input_upper, input_shape)
    }

    /// WGPU resident IBP is excluded from verdict authority while authenticated
    /// overflow-taint transport, terminal consultation, and the remaining
    /// general-authority obligations are incomplete.
    fn provides_sound_gpu_ibp(&self) -> bool {
        false
    }
}

impl GpuIbpForwardExt for WgpuDevice {
    fn prepare_model_plan(
        &self,
        layers: &[GpuIbpLayer],
        input_shape: &[usize],
    ) -> Result<Option<Box<dyn GpuIbpModelPlan>>> {
        if layers.is_empty() {
            return Ok(None);
        }

        Ok(Some(Box::new(
            self.prepare_model_plan_internal(layers, input_shape)?,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::super::ibp_forward_plan::max_resident_buffer_elems;
    use ny_core::GpuIbpLayer;
    use std::sync::Arc;

    #[test]
    fn test_max_resident_buffer_elems_tracks_batched_linear_outputs_4081() {
        let layers = vec![
            GpuIbpLayer::Linear {
                weight: Arc::from(vec![0.0_f32; 6]),
                bias: None,
                out_features: 3,
                in_features: 2,
            },
            GpuIbpLayer::ReLU { num_elements: 12 },
            GpuIbpLayer::Linear {
                weight: Arc::from(vec![0.0_f32; 15]),
                bias: None,
                out_features: 5,
                in_features: 3,
            },
        ];

        assert_eq!(
            max_resident_buffer_elems(&layers, 8).expect("batched chain should size buffers"),
            20
        );
    }

    #[test]
    fn test_max_resident_buffer_elems_rejects_view_shape_mismatch_4081() {
        let layers = vec![GpuIbpLayer::View {
            output_shape: Arc::from(vec![3usize, 3usize]),
        }];

        let err = max_resident_buffer_elems(&layers, 8)
            .expect_err("shape-changing view should fail closed");
        assert!(
            err.to_string().contains("expected"),
            "shape mismatch should be reported, got: {err}"
        );
    }

    #[test]
    fn test_max_resident_buffer_elems_tracks_conv2d_outputs_4275() {
        let layers = vec![
            GpuIbpLayer::Conv2d {
                weight: Arc::from(vec![0.0_f32; 8]),
                bias: Some(Arc::from(vec![0.0_f32; 2])),
                out_channels: 2,
                in_channels: 1,
                kernel_h: 2,
                kernel_w: 2,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
                groups: 1,
                input_h: 4,
                input_w: 4,
            },
            GpuIbpLayer::View {
                output_shape: Arc::from(vec![2usize, 18usize]),
            },
            GpuIbpLayer::Linear {
                weight: Arc::from(vec![0.0_f32; 54]),
                bias: None,
                out_features: 3,
                in_features: 18,
            },
        ];

        assert_eq!(
            max_resident_buffer_elems(&layers, 32).expect("conv chain should size buffers"),
            36
        );
    }
}
