// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::super::WgpuDevice;
use super::gpu_checked_u32;
use crate::wgpu_device::params::SoftmaxIbpParams;

impl WgpuDevice {
    /// Execute softmax IBP on the GPU.
    ///
    /// This is a two-pass operation:
    /// 1. Reduce pass: compute exp values and sums per row
    /// 2. Apply pass: compute final softmax bounds using Auto-LiRPA formula
    pub fn softmax_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Wrap GPU work in an error scope: a wgpu error returns Err (caller
        // falls back to CPU) instead of aborting via wgpu's panicking handler.
        self.run_gpu_checked("softmax_ibp", || self.softmax_ibp_inner(input))
    }

    fn softmax_ibp_inner(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(NyError::InvalidSpec("Empty input to softmax".to_string()));
        }

        // Softmax is along the last dimension
        let row_size = shape[shape.len() - 1];
        let row_dims = &shape[..shape.len() - 1];
        let num_rows: usize = checked_shape_product(row_dims).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "WgpuDevice softmax_ibp: row dims {row_dims:?} overflow usize",
            ))
        })?;
        let num_rows = if num_rows == 0 { 1 } else { num_rows };
        let total_elements = num_rows * row_size;

        debug!(
            "WgpuDevice softmax_ibp: num_rows={}, row_size={}",
            num_rows, row_size
        );

        // Get input data
        let input_lower = input.lower().as_slice().ok_or_else(|| {
            NyError::InternalError("wgpu softmax_ibp: input lower not contiguous".into())
        })?;
        let input_upper = input.upper().as_slice().ok_or_else(|| {
            NyError::InternalError("wgpu softmax_ibp: input upper not contiguous".into())
        })?;

        // Create params
        let params = SoftmaxIbpParams {
            num_rows: gpu_checked_u32(num_rows, "softmax num_rows")?,
            row_size: gpu_checked_u32(row_size, "softmax row_size")?,
            _padding: [0, 0],
        };

        // Get or create buffers from pool
        let mut pool = self.buffer_pool.lock().map_err(|e| {
            NyError::InternalError(format!("wgpu softmax_ibp: buffer pool lock poisoned: {e}"))
        })?;

        // Softmax params buffer
        if pool.softmax_params_buffer.is_none() {
            pool.softmax_params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("softmax_params_buffer"),
                size: size_of::<SoftmaxIbpParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let params_buffer = pool
            .softmax_params_buffer
            .as_ref()
            .ok_or_else(|| {
                NyError::InternalError("wgpu softmax_ibp: params buffer not created".into())
            })?
            .clone();

        // Input buffers (reuse from linear IBP)
        let input_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.input_lower_buffer,
            total_elements,
            "input_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let input_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.input_upper_buffer,
            total_elements,
            "input_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // Intermediate buffers
        let exp_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_exp_lower_buffer,
            total_elements,
            "exp_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let exp_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_exp_upper_buffer,
            total_elements,
            "exp_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let sum_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_sum_lower_buffer,
            num_rows,
            "sum_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let sum_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_sum_upper_buffer,
            num_rows,
            "sum_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let max_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_max_buffer,
            num_rows,
            "max_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // Output buffers
        let output_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.output_lower_buffer,
            total_elements,
            "output_lower_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let output_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.output_upper_buffer,
            total_elements,
            "output_upper_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Staging buffers
        let staging_lower = self.get_or_create_storage_buffer(
            &mut pool.staging_lower_buffer,
            total_elements,
            "staging_lower",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let staging_upper = self.get_or_create_storage_buffer(
            &mut pool.staging_upper_buffer,
            total_elements,
            "staging_upper",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Keep the shared buffer-pool lock alive until readback completes.
        // Otherwise concurrent Rayon threads can recycle staging buffers
        // while a previous `map_async`/`unmap` cycle is still in flight,
        // triggering wgpu's "Buffer ... is still mapped" validation panic (#3877).
        let _pool_guard = pool;

        // Write data to buffers
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[params]));
        self.queue
            .write_buffer(&input_lower_buffer, 0, bytemuck::cast_slice(input_lower));
        self.queue
            .write_buffer(&input_upper_buffer, 0, bytemuck::cast_slice(input_upper));

        // Create bind group for reduce pass
        let reduce_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_reduce_bind_group"),
            layout: &self.softmax_reduce_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: exp_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: exp_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: sum_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: sum_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: max_buffer.as_entire_binding(),
                },
            ],
        });

        // Create bind group for apply pass
        let apply_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_apply_bind_group"),
            layout: &self.softmax_apply_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: exp_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: exp_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sum_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sum_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output_upper_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("softmax_ibp_encoder"),
            });

        // Pass 1: Reduce (one thread per row)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_reduce_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.softmax_reduce_pipeline);
            compute_pass.set_bind_group(0, &reduce_bind_group, &[]);
            let workgroup_count =
                gpu_checked_u32(num_rows, "softmax reduce dispatch")?.div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Pass 2: Apply (one thread per element)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_apply_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.softmax_apply_pipeline);
            compute_pass.set_bind_group(0, &apply_bind_group, &[]);
            let workgroup_count =
                gpu_checked_u32(total_elements, "softmax apply dispatch")?.div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        let output_buffer_size = (total_elements * size_of::<f32>()) as u64;

        // Copy results to staging buffers
        encoder.copy_buffer_to_buffer(
            &output_lower_buffer,
            0,
            &staging_lower,
            0,
            output_buffer_size,
        );
        encoder.copy_buffer_to_buffer(
            &output_upper_buffer,
            0,
            &staging_upper,
            0,
            output_buffer_size,
        );

        // Submit and wait
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read results
        let mut result_lower = Self::read_buffer(&self.device, &staging_lower, total_elements)?;
        let mut result_upper = Self::read_buffer(&self.device, &staging_upper, total_elements)?;

        // Defense-in-depth: sanitize NaN/Inf from GPU readback (#2785).
        super::sanitize_readback(&mut result_lower, &mut result_upper);

        // Reshape to original shape
        let lower = ArrayD::from_shape_vec(IxDyn(shape), result_lower)
            .map_err(|_| NyError::shape_mismatch(shape.to_vec(), vec![total_elements]))?;
        let upper = ArrayD::from_shape_vec(IxDyn(shape), result_upper)
            .map_err(|_| NyError::shape_mismatch(shape.to_vec(), vec![total_elements]))?;

        BoundedTensor::new(lower, upper)
    }
}
