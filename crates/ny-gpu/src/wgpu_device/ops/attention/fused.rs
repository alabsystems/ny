// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::super::super::WgpuDevice;

use super::helpers::{array_slice, checked_mul_usize, to_u32};
use crate::wgpu_device::params::{
    MatmulIbpParams, ScaleIbpParams, SoftmaxIbpParams, TransposeIbpParams,
};

impl WgpuDevice {
    /// Fused attention IBP: Q @ K^T -> scale -> softmax -> probs @ V
    ///
    /// This chains all attention operations on the GPU without intermediate
    /// readbacks, providing significant speedup compared to separate calls.
    ///
    /// # Arguments
    /// * `q` - Query tensor with shape [batch, heads, seq, dim]
    /// * `k` - Key tensor with shape [batch, heads, seq, dim]
    /// * `v` - Value tensor with shape [batch, heads, seq, dim]
    /// * `scale` - Scaling factor (typically 1.0 / sqrt(dim))
    ///
    /// # Returns
    /// Output tensor with shape [batch, heads, seq, dim]
    pub fn attention_ibp_fused(
        &self,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
        scale: f32,
    ) -> Result<BoundedTensor> {
        // Wrap GPU work in an error scope: a wgpu error returns Err (caller
        // falls back to CPU) instead of aborting via wgpu's panicking handler.
        self.run_gpu_checked("attention_ibp_fused", || {
            self.attention_ibp_fused_inner(q, k, v, scale)
        })
    }

    fn attention_ibp_fused_inner(
        &self,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
        scale: f32,
    ) -> Result<BoundedTensor> {
        let shape_q = q.shape();
        let shape_k = k.shape();
        let shape_v = v.shape();

        // Validate shapes: all should be [batch, heads, seq, dim]
        if shape_q.len() != 4 || shape_k.len() != 4 || shape_v.len() != 4 {
            return Err(NyError::InvalidSpec(
                "Attention inputs must be 4D [batch, heads, seq, dim]".to_string(),
            ));
        }

        // Verify Q, K, V shapes are compatible
        if shape_q != shape_k {
            return Err(NyError::shape_mismatch(shape_q.to_vec(), shape_k.to_vec()));
        }
        if shape_q[..3] != shape_v[..3] || shape_v[3] != shape_q[3] {
            return Err(NyError::shape_mismatch(shape_q.to_vec(), shape_v.to_vec()));
        }

        let batch = shape_q[0];
        let heads = shape_q[1];
        let seq = shape_q[2];
        let dim = shape_q[3];
        let batch_heads = checked_mul_usize(batch, heads, "batch*heads")?;

        debug!(
            "WgpuDevice attention_ibp_fused: batch={}, heads={}, seq={}, dim={}, scale={}",
            batch, heads, seq, dim, scale
        );

        // Buffer sizes
        let qkv_size = checked_mul_usize(
            checked_mul_usize(batch_heads, seq, "batch_heads*seq")?,
            dim,
            "batch_heads*seq*dim",
        )?; // Q, K, V: [batch*heads, seq, dim]
        let kt_size = checked_mul_usize(
            checked_mul_usize(batch_heads, dim, "batch_heads*dim")?,
            seq,
            "batch_heads*dim*seq",
        )?; // K^T: [batch*heads, dim, seq]
        let scores_size = checked_mul_usize(
            checked_mul_usize(batch_heads, seq, "batch_heads*seq")?,
            seq,
            "batch_heads*seq*seq",
        )?; // scores: [batch*heads, seq, seq]
        let output_size = checked_mul_usize(
            checked_mul_usize(batch_heads, seq, "batch_heads*seq")?,
            dim,
            "batch_heads*seq*dim",
        )?; // output: [batch*heads, seq, dim]
        let softmax_rows = checked_mul_usize(batch_heads, seq, "batch_heads*seq")?; // rows

        let batch_heads_u32 = to_u32(batch_heads, "batch_heads")?;
        let seq_u32 = to_u32(seq, "seq")?;
        let dim_u32 = to_u32(dim, "dim")?;
        let kt_u32 = to_u32(kt_size, "kt_size")?;
        let scores_u32 = to_u32(scores_size, "scores_size")?;
        let output_u32 = to_u32(output_size, "output_size")?;
        let softmax_rows_u32 = to_u32(softmax_rows, "softmax_rows")?;

        // Get input data as contiguous slices
        let q_lower_data = array_slice("q.lower()", q.lower())?;
        let q_upper_data = array_slice("q.upper()", q.upper())?;
        let k_lower_data = array_slice("k.lower()", k.lower())?;
        let k_upper_data = array_slice("k.upper()", k.upper())?;
        let v_lower_data = array_slice("v.lower()", v.lower())?;
        let v_upper_data = array_slice("v.upper()", v.upper())?;

        // Get or create buffers from pool
        let mut pool = self.buffer_pool.lock().map_err(|e| {
            NyError::InternalError(format!(
                "wgpu attention_fused: buffer pool lock poisoned: {e}"
            ))
        })?;

        // === INPUT BUFFERS ===
        // Q buffers (reuse a_lower/a_upper from matmul)
        let q_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.a_lower_buffer,
            qkv_size,
            "q_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let q_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.a_upper_buffer,
            qkv_size,
            "q_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // K buffers (reuse input_lower/input_upper)
        let k_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.input_lower_buffer,
            qkv_size,
            "k_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let k_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.input_upper_buffer,
            qkv_size,
            "k_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // V buffers (reuse b_lower/b_upper from matmul)
        let v_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.b_lower_buffer,
            qkv_size,
            "v_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let v_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.b_upper_buffer,
            qkv_size,
            "v_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // === INTERMEDIATE BUFFERS ===
        // K^T buffers
        let kt_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.k_transposed_lower_buffer,
            kt_size,
            "kt_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let kt_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.k_transposed_upper_buffer,
            kt_size,
            "kt_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // QK scores buffers
        let qk_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.qk_scores_lower_buffer,
            scores_size,
            "qk_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let qk_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.qk_scores_upper_buffer,
            scores_size,
            "qk_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // Scaled scores buffers (reuse softmax exp buffers)
        let scaled_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_exp_lower_buffer,
            scores_size,
            "scaled_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let scaled_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_exp_upper_buffer,
            scores_size,
            "scaled_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // Softmax intermediate buffers
        let sum_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_sum_lower_buffer,
            softmax_rows,
            "sum_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let sum_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_sum_upper_buffer,
            softmax_rows,
            "sum_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let max_buffer = self.get_or_create_storage_buffer(
            &mut pool.softmax_max_buffer,
            softmax_rows,
            "max_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // Attention probs buffers
        let probs_lower_buffer = self.get_or_create_storage_buffer(
            &mut pool.attn_probs_lower_buffer,
            scores_size,
            "probs_lower_buffer",
            wgpu::BufferUsages::STORAGE,
        );
        let probs_upper_buffer = self.get_or_create_storage_buffer(
            &mut pool.attn_probs_upper_buffer,
            scores_size,
            "probs_upper_buffer",
            wgpu::BufferUsages::STORAGE,
        );

        // === OUTPUT BUFFERS ===
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

        // Staging buffers for final readback
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

        // === PARAMS BUFFERS ===
        // Transpose params
        if pool.transpose_params_buffer.is_none() {
            pool.transpose_params_buffer =
                Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("transpose_params_buffer"),
                    size: size_of::<TransposeIbpParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
        }
        let transpose_params_buffer = pool
            .transpose_params_buffer
            .as_ref()
            .ok_or_else(|| {
                NyError::InternalError("wgpu attention: transpose_params not created".into())
            })?
            .clone();

        // Matmul params (for Q @ K^T)
        if pool.matmul_params_buffer.is_none() {
            pool.matmul_params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("matmul_params_buffer"),
                size: size_of::<MatmulIbpParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let matmul_qk_params_buffer = pool
            .matmul_params_buffer
            .as_ref()
            .ok_or_else(|| {
                NyError::InternalError("wgpu attention: matmul_params not created".into())
            })?
            .clone();

        // Scale params
        if pool.scale_params_buffer.is_none() {
            pool.scale_params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scale_params_buffer"),
                size: size_of::<ScaleIbpParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let scale_params_buffer = pool
            .scale_params_buffer
            .as_ref()
            .ok_or_else(|| {
                NyError::InternalError("wgpu attention: scale_params not created".into())
            })?
            .clone();

        // Softmax params
        if pool.softmax_params_buffer.is_none() {
            pool.softmax_params_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("softmax_params_buffer"),
                size: size_of::<SoftmaxIbpParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let softmax_params_buffer = pool
            .softmax_params_buffer
            .as_ref()
            .ok_or_else(|| {
                NyError::InternalError("wgpu attention: softmax_params not created".into())
            })?
            .clone();

        // Second matmul params (for probs @ V, different from Q @ K^T)
        if pool.matmul_pv_params_buffer.is_none() {
            pool.matmul_pv_params_buffer =
                Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("matmul_pv_params_buffer"),
                    size: size_of::<MatmulIbpParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
        }
        let matmul_pv_params_buffer = pool
            .matmul_pv_params_buffer
            .as_ref()
            .ok_or_else(|| {
                NyError::InternalError("wgpu attention: matmul_pv_params not created".into())
            })?
            .clone();

        // Keep the shared buffer-pool lock alive until readback completes.
        // Otherwise concurrent Rayon threads can recycle staging buffers
        // while a previous `map_async`/`unmap` cycle is still in flight,
        // triggering wgpu's "Buffer ... is still mapped" validation panic (#3877).
        let _pool_guard = pool;

        // === WRITE DATA TO BUFFERS ===
        // Input data
        self.queue
            .write_buffer(&q_lower_buffer, 0, bytemuck::cast_slice(q_lower_data));
        self.queue
            .write_buffer(&q_upper_buffer, 0, bytemuck::cast_slice(q_upper_data));
        self.queue
            .write_buffer(&k_lower_buffer, 0, bytemuck::cast_slice(k_lower_data));
        self.queue
            .write_buffer(&k_upper_buffer, 0, bytemuck::cast_slice(k_upper_data));
        self.queue
            .write_buffer(&v_lower_buffer, 0, bytemuck::cast_slice(v_lower_data));
        self.queue
            .write_buffer(&v_upper_buffer, 0, bytemuck::cast_slice(v_upper_data));

        // Params
        let transpose_params = TransposeIbpParams {
            batch_size: batch_heads_u32,
            rows: seq_u32,
            cols: dim_u32,
            _padding: 0,
        };
        self.queue.write_buffer(
            &transpose_params_buffer,
            0,
            bytemuck::cast_slice(&[transpose_params]),
        );

        let matmul_qk_params = MatmulIbpParams {
            batch_size: batch_heads_u32,
            m: seq_u32, // rows of Q
            k: dim_u32, // cols of Q = rows of K^T
            n: seq_u32, // cols of K^T
        };
        self.queue.write_buffer(
            &matmul_qk_params_buffer,
            0,
            bytemuck::cast_slice(&[matmul_qk_params]),
        );

        let scale_params = ScaleIbpParams {
            total_elements: scores_u32,
            scale,
            _padding: [0, 0],
        };
        self.queue.write_buffer(
            &scale_params_buffer,
            0,
            bytemuck::cast_slice(&[scale_params]),
        );

        let softmax_params = SoftmaxIbpParams {
            num_rows: softmax_rows_u32,
            row_size: seq_u32,
            _padding: [0, 0],
        };
        self.queue.write_buffer(
            &softmax_params_buffer,
            0,
            bytemuck::cast_slice(&[softmax_params]),
        );

        // Matmul params for probs @ V (different dimensions from Q @ K^T)
        let matmul_pv_params = MatmulIbpParams {
            batch_size: batch_heads_u32,
            m: seq_u32, // rows of probs
            k: seq_u32, // cols of probs = rows of V
            n: dim_u32, // cols of V
        };
        self.queue.write_buffer(
            &matmul_pv_params_buffer,
            0,
            bytemuck::cast_slice(&[matmul_pv_params]),
        );

        // === CREATE BIND GROUPS ===
        // 1. Transpose K -> K^T
        let transpose_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transpose_bind_group"),
            layout: &self.transpose_ibp_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: transpose_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: k_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: kt_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: kt_upper_buffer.as_entire_binding(),
                },
            ],
        });

        // 2. Q @ K^T -> scores
        let matmul_qk_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_qk_bind_group"),
            layout: &self.matmul_ibp_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matmul_qk_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: q_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: q_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: kt_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: kt_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: qk_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: qk_upper_buffer.as_entire_binding(),
                },
            ],
        });

        // 3. Scale scores
        let scale_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scale_bind_group"),
            layout: &self.scale_ibp_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scale_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: qk_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: qk_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scaled_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scaled_upper_buffer.as_entire_binding(),
                },
            ],
        });

        // 4. Softmax reduce pass
        let softmax_reduce_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_reduce_bind_group_fused"),
            layout: &self.softmax_reduce_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: softmax_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scaled_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scaled_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: probs_lower_buffer.as_entire_binding(), // Reuse for exp_lower
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: probs_upper_buffer.as_entire_binding(), // Reuse for exp_upper
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

        // 5. Softmax apply pass (output to scaled buffers, will be input to final matmul)
        // We need separate buffers for apply output since probs_* has exp values from reduce
        let softmax_apply_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_apply_bind_group_fused"),
            layout: &self.softmax_apply_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: softmax_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: probs_lower_buffer.as_entire_binding(), // exp_lower from reduce
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: probs_upper_buffer.as_entire_binding(), // exp_upper from reduce
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
                    resource: qk_lower_buffer.as_entire_binding(), // Reuse for final probs
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: qk_upper_buffer.as_entire_binding(), // Reuse for final probs
                },
            ],
        });

        // 6. Final matmul: probs @ V -> output
        // Uses separate params buffer (matmul_pv_params_buffer) written earlier
        let matmul_pv_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_pv_bind_group"),
            layout: &self.matmul_ibp_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matmul_pv_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: qk_lower_buffer.as_entire_binding(), // Final probs (from softmax apply)
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: qk_upper_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v_lower_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: v_upper_buffer.as_entire_binding(),
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

        // === ENCODE ALL OPERATIONS ===
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("attention_ibp_fused_encoder"),
            });

        // Pass 1: Transpose K -> K^T
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("transpose_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.transpose_ibp_pipeline);
            compute_pass.set_bind_group(0, &transpose_bind_group, &[]);
            let workgroup_count = kt_u32.div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Pass 2: Q @ K^T -> scores
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_qk_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.matmul_ibp_pipeline);
            compute_pass.set_bind_group(0, &matmul_qk_bind_group, &[]);
            let workgroup_count = scores_u32.div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Pass 3: Scale scores
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scale_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.scale_ibp_pipeline);
            compute_pass.set_bind_group(0, &scale_bind_group, &[]);
            let workgroup_count = scores_u32.div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Pass 4: Softmax reduce
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_reduce_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.softmax_reduce_pipeline);
            compute_pass.set_bind_group(0, &softmax_reduce_bind_group, &[]);
            let workgroup_count = softmax_rows_u32.div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Pass 5: Softmax apply
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_apply_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.softmax_apply_pipeline);
            compute_pass.set_bind_group(0, &softmax_apply_bind_group, &[]);
            let workgroup_count = scores_u32.div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Pass 6: probs @ V -> output (uses matmul_pv_params_buffer written earlier)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pv_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.matmul_ibp_pipeline);
            compute_pass.set_bind_group(0, &matmul_pv_bind_group, &[]);
            let workgroup_count = output_u32.div_ceil(64);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy results to staging buffers
        let output_buffer_size = (output_size * size_of::<f32>()) as u64;
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

        // Submit all passes in one command buffer
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back final results only
        let mut result_lower = Self::read_buffer(&self.device, &staging_lower, output_size)?;
        let mut result_upper = Self::read_buffer(&self.device, &staging_upper, output_size)?;

        // Defense-in-depth: sanitize NaN/Inf from GPU readback (#2785).
        super::super::sanitize_readback(&mut result_lower, &mut result_upper);

        // Build output shape: [batch, heads, seq, dim]
        let out_shape = vec![batch, heads, seq, dim];
        let lower = ArrayD::from_shape_vec(IxDyn(&out_shape), result_lower)
            .map_err(|_| NyError::shape_mismatch(out_shape.clone(), vec![output_size]))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&out_shape), result_upper)
            .map_err(|_| NyError::shape_mismatch(out_shape, vec![output_size]))?;

        BoundedTensor::new(lower, upper)
    }
}
