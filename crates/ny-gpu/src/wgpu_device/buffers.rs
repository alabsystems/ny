// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::WgpuDevice;

/// Pool of reusable GPU buffers.
///
/// Buffers are grown as needed but never shrunk, avoiding repeated allocations
/// for operations with similar or smaller sizes.
#[derive(Default)]
pub(super) struct BufferPool {
    // Linear IBP buffers
    /// Params uniform buffer for linear IBP (fixed size)
    pub(super) linear_params_buffer: Option<wgpu::Buffer>,
    /// Input lower bounds storage buffer
    pub(super) input_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// Input upper bounds storage buffer
    pub(super) input_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// Weight positive parts storage buffer
    pub(super) weight_pos_buffer: Option<(wgpu::Buffer, usize)>,
    /// Weight negative parts storage buffer
    pub(super) weight_neg_buffer: Option<(wgpu::Buffer, usize)>,
    /// Bias storage buffer
    pub(super) bias_buffer: Option<(wgpu::Buffer, usize)>,
    /// Output lower bounds storage buffer (shared with matmul)
    pub(super) output_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// Output upper bounds storage buffer (shared with matmul)
    pub(super) output_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// Staging buffer for output lower bounds readback (shared with matmul)
    pub(super) staging_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// Staging buffer for output upper bounds readback (shared with matmul)
    pub(super) staging_upper_buffer: Option<(wgpu::Buffer, usize)>,

    // MatMul IBP buffers
    /// Params uniform buffer for matmul IBP (fixed size)
    pub(super) matmul_params_buffer: Option<wgpu::Buffer>,
    /// A lower bounds storage buffer
    pub(super) a_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// A upper bounds storage buffer
    pub(super) a_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// B lower bounds storage buffer
    pub(super) b_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// B upper bounds storage buffer
    pub(super) b_upper_buffer: Option<(wgpu::Buffer, usize)>,

    // Softmax IBP buffers
    /// Params uniform buffer for softmax IBP (fixed size)
    pub(super) softmax_params_buffer: Option<wgpu::Buffer>,
    /// Softmax intermediate: exp_lower
    pub(super) softmax_exp_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// Softmax intermediate: exp_upper
    pub(super) softmax_exp_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// Softmax intermediate: sum_exp_lower per row
    pub(super) softmax_sum_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// Softmax intermediate: sum_exp_upper per row
    pub(super) softmax_sum_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// Softmax intermediate: max_upper per row
    pub(super) softmax_max_buffer: Option<(wgpu::Buffer, usize)>,

    // Fused attention buffers - keep intermediate results on GPU
    /// Params uniform buffer for transpose IBP (fixed size)
    pub(super) transpose_params_buffer: Option<wgpu::Buffer>,
    /// Params uniform buffer for scale IBP (fixed size)
    pub(super) scale_params_buffer: Option<wgpu::Buffer>,
    /// K transposed buffer (intermediate for attention)
    pub(super) k_transposed_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// K transposed upper buffer (intermediate for attention)
    pub(super) k_transposed_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// QK scores buffer (intermediate for attention)
    pub(super) qk_scores_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// QK scores upper buffer (intermediate for attention)
    pub(super) qk_scores_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// Attention probs buffer (intermediate for attention)
    pub(super) attn_probs_lower_buffer: Option<(wgpu::Buffer, usize)>,
    /// Attention probs upper buffer (intermediate for attention)
    pub(super) attn_probs_upper_buffer: Option<(wgpu::Buffer, usize)>,
    /// Second matmul params buffer (for probs@V in fused attention)
    pub(super) matmul_pv_params_buffer: Option<wgpu::Buffer>,

    // GEMM buffers (CROWN linear backward)
    /// Params uniform buffer for GEMM (fixed size)
    pub(super) gemm_params_buffer: Option<wgpu::Buffer>,
    /// A storage buffer
    pub(super) gemm_a_buffer: Option<(wgpu::Buffer, usize)>,
    /// B storage buffer
    pub(super) gemm_b_buffer: Option<(wgpu::Buffer, usize)>,
    /// Output storage buffer
    pub(super) gemm_out_buffer: Option<(wgpu::Buffer, usize)>,
    /// Staging buffer for output readback
    pub(super) gemm_staging_buffer: Option<(wgpu::Buffer, usize)>,

    // CROWN backward buffers (persistent GPU A-matrices, #3397)
    /// Params uniform buffer for CROWN backward shaders (fixed size)
    pub(super) crown_params_buffer: Option<wgpu::Buffer>,
    /// A-matrix lower bounds — ping buffer (num_specs × max_dim)
    pub(super) crown_a_lower_0: Option<(wgpu::Buffer, usize)>,
    /// A-matrix upper bounds — ping buffer
    pub(super) crown_a_upper_0: Option<(wgpu::Buffer, usize)>,
    /// A-matrix lower bounds — pong buffer
    pub(super) crown_a_lower_1: Option<(wgpu::Buffer, usize)>,
    /// A-matrix upper bounds — pong buffer
    pub(super) crown_a_upper_1: Option<(wgpu::Buffer, usize)>,
    /// Running bias lower accumulator (num_specs)
    pub(super) crown_bias_lower: Option<(wgpu::Buffer, usize)>,
    /// Running bias upper accumulator (num_specs)
    pub(super) crown_bias_upper: Option<(wgpu::Buffer, usize)>,
    /// Per-neuron activation slopes/intercepts upload buffer
    pub(super) crown_slopes_buffer: Option<(wgpu::Buffer, usize)>,
    /// Per-layer weight matrix upload buffer
    pub(super) crown_weight_buffer: Option<(wgpu::Buffer, usize)>,
    /// Per-layer bias vector upload buffer
    pub(super) crown_layer_bias_buffer: Option<(wgpu::Buffer, usize)>,
    /// Input lower bounds for concretization
    pub(super) crown_input_lower: Option<(wgpu::Buffer, usize)>,
    /// Input upper bounds for concretization
    pub(super) crown_input_upper: Option<(wgpu::Buffer, usize)>,
    /// Concretized output lower bounds (num_specs)
    pub(super) crown_output_lower: Option<(wgpu::Buffer, usize)>,
    /// Concretized output upper bounds (num_specs)
    pub(super) crown_output_upper: Option<(wgpu::Buffer, usize)>,
    /// Staging buffer for concretized output readback
    pub(super) crown_staging_lower: Option<(wgpu::Buffer, usize)>,
    /// Staging buffer for concretized output readback
    pub(super) crown_staging_upper: Option<(wgpu::Buffer, usize)>,

    // Conv2d CROWN backward intermediate buffers (#3397)
    /// Reshaped A-matrix lower: (total_spatial, out_c)
    pub(super) conv_reshaped_lower: Option<(wgpu::Buffer, usize)>,
    /// Reshaped A-matrix upper: (total_spatial, out_c)
    pub(super) conv_reshaped_upper: Option<(wgpu::Buffer, usize)>,
    /// GEMM output lower: (total_spatial, kernel_cols)
    pub(super) conv_gemm_out_lower: Option<(wgpu::Buffer, usize)>,
    /// GEMM output upper: (total_spatial, kernel_cols)
    pub(super) conv_gemm_out_upper: Option<(wgpu::Buffer, usize)>,
}

impl BufferPool {
    /// Release all CROWN backward buffer slots, returning memory to the system.
    ///
    /// Call between models in the VNN-COMP runner to prevent cross-model memory
    /// accumulation from the grow-only buffer pool (#3515).
    // Used in tests and Phase 2 VNN-COMP runner integration.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn release_crown_buffers(&mut self) {
        self.crown_params_buffer = None;
        self.crown_a_lower_0 = None;
        self.crown_a_upper_0 = None;
        self.crown_a_lower_1 = None;
        self.crown_a_upper_1 = None;
        self.crown_bias_lower = None;
        self.crown_bias_upper = None;
        self.crown_slopes_buffer = None;
        self.crown_weight_buffer = None;
        self.crown_layer_bias_buffer = None;
        self.crown_input_lower = None;
        self.crown_input_upper = None;
        self.crown_output_lower = None;
        self.crown_output_upper = None;
        self.crown_staging_lower = None;
        self.crown_staging_upper = None;
        self.conv_reshaped_lower = None;
        self.conv_reshaped_upper = None;
        self.conv_gemm_out_lower = None;
        self.conv_gemm_out_upper = None;
    }
}

impl WgpuDevice {
    /// Get or create a storage buffer, reusing from pool if possible.
    pub(super) fn get_or_create_storage_buffer(
        &self,
        pool_slot: &mut Option<(wgpu::Buffer, usize)>,
        required_size: usize,
        label: &str,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        // Check if existing buffer is large enough
        if let Some((ref buffer, size)) = pool_slot {
            if *size >= required_size {
                return buffer.clone();
            }
        }

        // Create new buffer with 20% growth factor to avoid repeated resizing
        let new_size = (required_size as f64 * 1.2).ceil() as usize;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (new_size * size_of::<f32>()) as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        *pool_slot = Some((buffer.clone(), new_size));
        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify release_crown_buffers clears all CROWN-related pool slots (#3515).
    #[test]
    fn test_release_crown_buffers_clears_all_slots() {
        let mut pool = BufferPool::default();
        // All crown slots start as None; release should be safe on empty pool
        pool.release_crown_buffers();
        assert!(pool.crown_params_buffer.is_none());
        assert!(pool.crown_a_lower_0.is_none());
        assert!(pool.conv_gemm_out_upper.is_none());
    }
}
