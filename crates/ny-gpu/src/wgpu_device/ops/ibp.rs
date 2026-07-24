// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{nan_propagating_max_zero, nan_propagating_min_zero, NyError, Result};
use ny_tensor::BoundedTensor;

use super::super::WgpuDevice;
use super::gpu_checked_u32;
use crate::wgpu_device::params::{LinearIbpParams, MatmulIbpParams};

impl WgpuDevice {
    /// Execute linear IBP on the GPU with buffer reuse.
    pub(super) fn execute_linear_ibp(
        &self,
        input: &BoundedTensor,
        weight: &Array2<f32>,
        bias: Option<&Array1<f32>>,
        batch_size: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<BoundedTensor> {
        let shape = input.shape();

        // Pre-compute positive and negative weight matrices.
        // NaN-propagating variants (#2642): Rust's f32::max(NaN, 0.0) = 0.0 (IEEE 754-2008),
        // silently dropping NaN weights. Use nan_propagating_max_zero/min_zero so NaN weights
        // poison the accumulator and trigger downstream NaN guards, matching the CPU path (#2415).
        let weight_pos: Vec<f32> = weight
            .iter()
            .map(|&w| nan_propagating_max_zero(w))
            .collect();
        let weight_neg: Vec<f32> = weight
            .iter()
            .map(|&w| nan_propagating_min_zero(w))
            .collect();

        // Get input data
        let input_lower = input.lower().as_slice().ok_or_else(|| {
            NyError::InternalError("wgpu linear_ibp: input lower not contiguous".into())
        })?;
        let input_upper = input.upper().as_slice().ok_or_else(|| {
            NyError::InternalError("wgpu linear_ibp: input upper not contiguous".into())
        })?;

        // Bias (zeros if not provided)
        let bias_data: Vec<f32> = match bias {
            Some(b) => b
                .as_slice()
                .ok_or_else(|| {
                    NyError::InternalError("wgpu linear_ibp: bias not contiguous".into())
                })?
                .to_vec(),
            None => vec![0.0; out_features],
        };

        // Create params
        let params = LinearIbpParams {
            batch_size: gpu_checked_u32(batch_size, "linear_ibp batch_size")?,
            in_features: gpu_checked_u32(in_features, "linear_ibp in_features")?,
            out_features: gpu_checked_u32(out_features, "linear_ibp out_features")?,
            _padding: 0,
        };

        let input_size = batch_size * in_features;
        let weight_size = in_features * out_features;
        let output_size = batch_size * out_features;

        // Get or create buffers from pool
        let mut pool = self.buffer_pool.lock().map_err(|e| {
            NyError::InternalError(format!("wgpu linear_ibp: buffer pool lock poisoned: {e}"))
        })?;

        // Params buffer (fixed size, create once)
        if pool.linear_params_buffer.is_none() {
            pool.linear_params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("linear_params_buffer"),
                size: size_of::<LinearIbpParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let params_buffer = pool
            .linear_params_buffer
            .as_ref()
            .ok_or_else(|| {
                NyError::InternalError("wgpu linear_ibp: params buffer not created".into())
            })?
            .clone();

        // Get or resize storage buffers
        let input_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.input_lower_buffer,
            input_size,
            "input_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let input_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.input_upper_buffer,
            input_size,
            "input_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let weight_pos_buffer = self.get_or_create_storage_buffer(
            &mut pool.weight_pos_buffer,
            weight_size,
            "weight_pos_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let weight_neg_buffer = self.get_or_create_storage_buffer(
            &mut pool.weight_neg_buffer,
            weight_size,
            "weight_neg_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let bias_buffer = self.get_or_create_storage_buffer(
            &mut pool.bias_buffer,
            out_features,
            "bias_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let output_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.output_lower_buffer,
            output_size,
            "output_lower_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let output_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.output_upper_buffer,
            output_size,
            "output_upper_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Staging buffers for readback
        let staging_lower = self.get_or_create_storage_buffer(
            &mut pool.staging_lower_buffer,
            output_size,
            "staging_lower",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let staging_upper = self.get_or_create_storage_buffer(
            &mut pool.staging_upper_buffer,
            output_size,
            "staging_upper",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Hold pool lock through readback to prevent staging-buffer race (#3877).
        let _pool_guard = pool;

        // Write data to buffers using queue.write_buffer (avoids buffer creation)
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[params]));
        self.queue
            .write_buffer(&input_lower_buffer, 0, bytemuck::cast_slice(input_lower));
        self.queue
            .write_buffer(&input_upper_buffer, 0, bytemuck::cast_slice(input_upper));
        self.queue
            .write_buffer(&weight_pos_buffer, 0, bytemuck::cast_slice(&weight_pos));
        self.queue
            .write_buffer(&weight_neg_buffer, 0, bytemuck::cast_slice(&weight_neg));
        self.queue
            .write_buffer(&bias_buffer, 0, bytemuck::cast_slice(&bias_data));

        // Create bind group (must be created each call since buffers may have changed)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("linear_ibp_bind_group"),
            layout: &self.linear_ibp_bind_group_layout,
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
                    resource: weight_pos_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: weight_neg_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: output_upper_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("linear_ibp_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("linear_ibp_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.linear_ibp_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups: one thread per (batch, output_feature) pair
            // Workgroup size is 64 threads (defined in shader)
            let workgroup_count =
                gpu_checked_u32(batch_size * out_features, "linear_ibp dispatch")?.div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        let output_buffer_size = (output_size * size_of::<f32>()) as u64;

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
        let mut result_lower = Self::read_buffer(&self.device, &staging_lower, output_size)?;
        let mut result_upper = Self::read_buffer(&self.device, &staging_upper, output_size)?;

        // Defense-in-depth: sanitize NaN/Inf from GPU readback (#2785).
        super::sanitize_readback(&mut result_lower, &mut result_upper);

        // Reshape to output shape [..., out_features]
        let mut out_shape = shape[..shape.len() - 1].to_vec();
        out_shape.push(out_features);

        let lower = ArrayD::from_shape_vec(IxDyn(&out_shape), result_lower)
            .map_err(|_| NyError::shape_mismatch(out_shape.clone(), vec![output_size]))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&out_shape), result_upper)
            .map_err(|_| NyError::shape_mismatch(out_shape, vec![output_size]))?;

        BoundedTensor::new(lower, upper)
    }

    /// Execute matmul IBP on the GPU with buffer reuse.
    // Justification: GPU matmul IBP requires both input bound tensors, batch size,
    // matrix dimensions (m, k, n), and batch dimension layout for kernel dispatch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_matmul_ibp(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
        batch_size: usize,
        m: usize,
        k: usize,
        n: usize,
        batch_dims: &[usize],
    ) -> Result<BoundedTensor> {
        // Get input data
        let a_lower_data = input_a.lower().as_slice().ok_or_else(|| {
            NyError::InternalError("wgpu matmul_ibp: input_a lower not contiguous".into())
        })?;
        let a_upper_data = input_a.upper().as_slice().ok_or_else(|| {
            NyError::InternalError("wgpu matmul_ibp: input_a upper not contiguous".into())
        })?;
        let b_lower_data = input_b.lower().as_slice().ok_or_else(|| {
            NyError::InternalError("wgpu matmul_ibp: input_b lower not contiguous".into())
        })?;
        let b_upper_data = input_b.upper().as_slice().ok_or_else(|| {
            NyError::InternalError("wgpu matmul_ibp: input_b upper not contiguous".into())
        })?;

        // Create params
        let params = MatmulIbpParams {
            batch_size: gpu_checked_u32(batch_size, "matmul_ibp batch_size")?,
            m: gpu_checked_u32(m, "matmul_ibp m")?,
            k: gpu_checked_u32(k, "matmul_ibp k")?,
            n: gpu_checked_u32(n, "matmul_ibp n")?,
        };

        let a_size = batch_size * m * k;
        let b_size = batch_size * k * n;
        let output_size = batch_size * m * n;

        // Get or create buffers from pool
        let mut pool = self.buffer_pool.lock().map_err(|e| {
            NyError::InternalError(format!("wgpu matmul_ibp: buffer pool lock poisoned: {e}"))
        })?;

        // Matmul params buffer
        if pool.matmul_params_buffer.is_none() {
            pool.matmul_params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("matmul_params_buffer"),
                size: size_of::<MatmulIbpParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let params_buffer = pool
            .matmul_params_buffer
            .as_ref()
            .ok_or_else(|| {
                NyError::InternalError("wgpu matmul_ibp: params buffer not created".into())
            })?
            .clone();

        // Get or resize storage buffers
        let a_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.a_lower_buffer,
            a_size,
            "a_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let a_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.a_upper_buffer,
            a_size,
            "a_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let b_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.b_lower_buffer,
            b_size,
            "b_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let b_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.b_upper_buffer,
            b_size,
            "b_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // Reuse output buffers from linear IBP
        let output_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.output_lower_buffer,
            output_size,
            "output_lower_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let output_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.output_upper_buffer,
            output_size,
            "output_upper_buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Staging buffers for readback
        let staging_lower = self.get_or_create_storage_buffer(
            &mut pool.staging_lower_buffer,
            output_size,
            "staging_lower",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let staging_upper = self.get_or_create_storage_buffer(
            &mut pool.staging_upper_buffer,
            output_size,
            "staging_upper",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Hold pool lock through readback to prevent staging-buffer race (#3877).
        let _pool_guard = pool;

        // Write data to buffers
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[params]));
        self.queue
            .write_buffer(&a_lower_buffer, 0, bytemuck::cast_slice(a_lower_data));
        self.queue
            .write_buffer(&a_upper_buffer, 0, bytemuck::cast_slice(a_upper_data));
        self.queue
            .write_buffer(&b_lower_buffer, 0, bytemuck::cast_slice(b_lower_data));
        self.queue
            .write_buffer(&b_upper_buffer, 0, bytemuck::cast_slice(b_upper_data));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_ibp_bind_group"),
            layout: &self.matmul_ibp_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: a_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_upper_buffer.as_entire_binding(),
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

        // Create command encoder and dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_ibp_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_ibp_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.matmul_ibp_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: one thread per output element
            let workgroup_count = gpu_checked_u32(output_size, "matmul_ibp dispatch")?.div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        let output_buffer_size = (output_size * size_of::<f32>()) as u64;

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
        let mut result_lower = Self::read_buffer(&self.device, &staging_lower, output_size)?;
        let mut result_upper = Self::read_buffer(&self.device, &staging_upper, output_size)?;

        // Defense-in-depth: sanitize NaN/Inf from GPU readback (#2785).
        super::sanitize_readback(&mut result_lower, &mut result_upper);

        // Build output shape: [...batch_dims, m, n]
        let mut out_shape = batch_dims.to_vec();
        out_shape.push(m);
        out_shape.push(n);

        let lower = ArrayD::from_shape_vec(IxDyn(&out_shape), result_lower)
            .map_err(|_| NyError::shape_mismatch(out_shape.clone(), vec![output_size]))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&out_shape), result_upper)
            .map_err(|_| NyError::shape_mismatch(out_shape, vec![output_size]))?;

        BoundedTensor::new(lower, upper)
    }
}
