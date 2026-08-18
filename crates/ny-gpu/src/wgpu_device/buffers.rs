// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::WgpuDevice;
use ny_core::{NyError, Result};

/// Pool of reusable GPU buffers.
///
/// Buffers are grown as needed but never shrunk, avoiding repeated allocations
/// for operations with similar or smaller sizes.
#[derive(Default)]
// GEMM buffers are retained for `gpu-tests` diagnostics while the public proof
// adapter is quarantined. They are intentionally dormant in release builds.
#[allow(dead_code)]
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
    /// Exact logical bytes owned by every currently populated pool slot.
    ///
    /// Use `Buffer::size()` rather than the parallel element-capacity fields so
    /// bare uniform slots and any API-level allocation rounding are included.
    fn retained_device_bytes(&self) -> Result<usize> {
        fn add(total: &mut usize, buffer: &wgpu::Buffer, label: &str) -> Result<()> {
            let bytes = usize::try_from(buffer.size()).map_err(|_| {
                NyError::InternalError(format!("retained buffer `{label}` does not fit in usize"))
            })?;
            *total = total.checked_add(bytes).ok_or_else(|| {
                NyError::InternalError("retained buffer-pool byte count overflow".into())
            })?;
            Ok(())
        }

        let mut total = 0usize;
        macro_rules! add_bare {
            ($field:ident) => {
                if let Some(buffer) = self.$field.as_ref() {
                    add(&mut total, buffer, stringify!($field))?;
                }
            };
        }
        macro_rules! add_sized {
            ($field:ident) => {
                if let Some((buffer, _)) = self.$field.as_ref() {
                    add(&mut total, buffer, stringify!($field))?;
                }
            };
        }

        add_bare!(linear_params_buffer);
        add_sized!(input_lower_buffer);
        add_sized!(input_upper_buffer);
        add_sized!(weight_pos_buffer);
        add_sized!(weight_neg_buffer);
        add_sized!(bias_buffer);
        add_sized!(output_lower_buffer);
        add_sized!(output_upper_buffer);
        add_sized!(staging_lower_buffer);
        add_sized!(staging_upper_buffer);

        add_bare!(matmul_params_buffer);
        add_sized!(a_lower_buffer);
        add_sized!(a_upper_buffer);
        add_sized!(b_lower_buffer);
        add_sized!(b_upper_buffer);

        add_bare!(softmax_params_buffer);
        add_sized!(softmax_exp_lower_buffer);
        add_sized!(softmax_exp_upper_buffer);
        add_sized!(softmax_sum_lower_buffer);
        add_sized!(softmax_sum_upper_buffer);
        add_sized!(softmax_max_buffer);

        add_bare!(transpose_params_buffer);
        add_bare!(scale_params_buffer);
        add_sized!(k_transposed_lower_buffer);
        add_sized!(k_transposed_upper_buffer);
        add_sized!(qk_scores_lower_buffer);
        add_sized!(qk_scores_upper_buffer);
        add_sized!(attn_probs_lower_buffer);
        add_sized!(attn_probs_upper_buffer);
        add_bare!(matmul_pv_params_buffer);

        add_bare!(gemm_params_buffer);
        add_sized!(gemm_a_buffer);
        add_sized!(gemm_b_buffer);
        add_sized!(gemm_out_buffer);
        add_sized!(gemm_staging_buffer);

        add_bare!(crown_params_buffer);
        add_sized!(crown_a_lower_0);
        add_sized!(crown_a_upper_0);
        add_sized!(crown_a_lower_1);
        add_sized!(crown_a_upper_1);
        add_sized!(crown_bias_lower);
        add_sized!(crown_bias_upper);
        add_sized!(crown_slopes_buffer);
        add_sized!(crown_weight_buffer);
        add_sized!(crown_layer_bias_buffer);
        add_sized!(crown_input_lower);
        add_sized!(crown_input_upper);
        add_sized!(crown_output_lower);
        add_sized!(crown_output_upper);
        add_sized!(crown_staging_lower);
        add_sized!(crown_staging_upper);
        add_sized!(conv_reshaped_lower);
        add_sized!(conv_reshaped_upper);
        add_sized!(conv_gemm_out_lower);
        add_sized!(conv_gemm_out_upper);
        Ok(total)
    }

    /// Release all CROWN backward buffer slots, returning memory to the system.
    ///
    /// Call between models in the VNN-COMP runner to prevent cross-model memory
    /// accumulation from the grow-only buffer pool (#3515).
    // Used in tests and Phase 2 VNN-COMP runner integration.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn release_crown_buffers(&mut self) {
        // #gpu-pool-highwater: keep the ledger honest about what this drops, so
        // `live_bytes()` reflects the release rather than only ever rising.
        // `crown_params_buffer` is a bare `Option<Buffer>` (fixed-size uniform,
        // never sized by the workload) so it is not tracked here.
        for slot in [
            &self.crown_a_lower_0,
            &self.crown_a_upper_0,
            &self.crown_a_lower_1,
            &self.crown_a_upper_1,
            &self.crown_bias_lower,
            &self.crown_bias_upper,
            &self.crown_slopes_buffer,
            &self.crown_weight_buffer,
            &self.crown_layer_bias_buffer,
            &self.crown_input_lower,
            &self.crown_input_upper,
            &self.crown_output_lower,
            &self.crown_output_upper,
            &self.crown_staging_lower,
            &self.crown_staging_upper,
            &self.conv_reshaped_lower,
            &self.conv_reshaped_upper,
            &self.conv_gemm_out_lower,
            &self.conv_gemm_out_upper,
        ] {
            if let Some((_, elems)) = slot.as_ref() {
                crate::gpu_memory_ledger::record_free((*elems * size_of::<f32>()) as u64);
            }
        }

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
    /// Checked retained bytes in this exact device's reusable buffer pool.
    pub(crate) fn buffer_pool_retained_bytes(&self) -> Result<usize> {
        let pool = self.buffer_pool.lock().map_err(|err| {
            NyError::InternalError(format!("buffer-pool retention lock poisoned: {err}"))
        })?;
        pool.retained_device_bytes()
    }

    /// Get or create a storage buffer, reusing from pool if possible.
    pub(super) fn get_or_create_storage_buffer(
        &self,
        pool_slot: &mut Option<(wgpu::Buffer, usize)>,
        required_size: usize,
        label: &str,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        // Check if existing buffer is large enough.
        //
        // #gpu-pool-highwater: "large enough" used to be the ONLY test, which
        // made every slot a permanent high-water mark -- a slot that once held
        // a multi-GiB buffer kept it for the life of the process even if every
        // later request was a few MiB, and the only release
        // (`release_crown_buffers`) is reachable solely through the public
        // `clear_crown_working_set()` "for long-lived runners", which nothing
        // calls during a run. On Apple silicon those bytes are host RAM.
        //
        // So also refuse a cached buffer that is MUCH larger than needed, and
        // reallocate at the smaller size. The ratio is what keeps this from
        // becoming a realloc treadmill: a slot whose demand genuinely
        // oscillates within 4x still reuses on every call, exactly as before.
        if let Some((ref buffer, size)) = pool_slot {
            if *size >= required_size && !oversized(*size, required_size) {
                return buffer.clone();
            }
        }

        // Create new buffer with 20% growth factor to avoid repeated resizing
        let new_size = (required_size as f64 * 1.2).ceil() as usize;
        let new_bytes = (new_size * size_of::<f32>()) as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: new_bytes,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Account for the slot's replacement: the outgoing buffer is dropped
        // when the slot is overwritten below.
        if let Some((_, old_size)) = pool_slot.as_ref() {
            crate::gpu_memory_ledger::record_free((*old_size * size_of::<f32>()) as u64);
        }
        crate::gpu_memory_ledger::record_alloc(label, new_bytes);

        *pool_slot = Some((buffer.clone(), new_size));
        buffer
    }
}

/// Retention ratio above which a pooled buffer is reallocated smaller rather
/// than reused (#gpu-pool-highwater).
///
/// 4x is deliberately loose. The pool exists to stop per-dispatch reallocation,
/// and CROWN buffer demand legitimately varies run to run with spec count and
/// layer width, so a tight ratio would trade the memory back for allocation
/// churn on the hot path. 4x only fires on the pathological case this guard is
/// for: a slot sized by one outlier workload and then never asked for anything
/// near that size again.
const POOL_SHRINK_RATIO: usize = 4;

/// Whether a cached buffer is so much larger than the request that holding it
/// costs more than reallocating (#gpu-pool-highwater).
///
/// `NY_GPU_POOL_SHRINK=0` restores the pre-2026-07-31 behavior of reusing any
/// buffer that merely fits, for A/B measurement and as a kill switch.
fn oversized(cached_size: usize, required_size: usize) -> bool {
    if !pool_shrink_enabled() {
        return false;
    }
    // Never churn on tiny buffers: below a slab's worth the reallocation is
    // pure overhead, and Metal suballocates from 128 MiB slabs anyway so
    // shrinking within one buys no resident bytes back.
    const MIN_SHRINK_ELEMS: usize = 4 * 1024 * 1024 / size_of::<f32>();
    cached_size > MIN_SHRINK_ELEMS && required_size.saturating_mul(POOL_SHRINK_RATIO) < cached_size
}

/// Kill switch for the shrink policy (`NY_GPU_POOL_SHRINK=0`).
fn pool_shrink_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NY_GPU_POOL_SHRINK").ok().as_deref() != Some("0"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #gpu-pool-highwater: a slot must reuse a buffer that merely fits, and
    /// must REFUSE one that is pathologically larger than the request. Without
    /// the second half the slot is a permanent high-water mark.
    #[test]
    fn oversized_refuses_only_pathological_retention() {
        let mib = 1024 * 1024 / size_of::<f32>();
        // Exact fit and modest slack: always reuse.
        assert!(!oversized(100 * mib, 100 * mib));
        assert!(!oversized(100 * mib, 50 * mib), "2x slack must still reuse");
        assert!(
            !oversized(100 * mib, 25 * mib + 1),
            "just under 4x must still reuse"
        );
        // Beyond the ratio: reallocate smaller.
        assert!(
            oversized(100 * mib, 10 * mib),
            "10x retention is what this guard exists for"
        );
    }

    /// Small buffers must never churn: below the floor, reallocating buys no
    /// resident bytes back (Metal suballocates from 128 MiB slabs) and only
    /// costs allocations on the hot path.
    #[test]
    fn oversized_never_fires_below_the_small_buffer_floor() {
        assert!(
            !oversized(1024, 1),
            "a 4 KiB cached buffer must not be reallocated however small the request"
        );
        let mib = 1024 * 1024 / size_of::<f32>();
        assert!(!oversized(mib, 1), "1 MiB is below the shrink floor");
    }

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
