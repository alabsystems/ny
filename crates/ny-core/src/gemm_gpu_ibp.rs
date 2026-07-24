// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::{NyError, Result};

/// Result from GPU-resident IBP forward pass.
///
/// Returns flattened lower and upper bounds for the network output,
/// along with the output shape for reconstruction into `BoundedTensor`.
pub struct GpuIbpResult {
    /// Lower bounds for each output element
    pub lower_bounds: Vec<f32>,
    /// Upper bounds for each output element
    pub upper_bounds: Vec<f32>,
    /// Logical output shape (e.g., `[out_features]` for a dense network)
    pub output_shape: Vec<usize>,
}

/// Per-layer descriptor for GPU-resident IBP forward pass.
///
/// Describes one layer in the forward propagation sequence. The first
/// resident packet supports dense and conv chains: `Linear`, `Conv2d`,
/// `ReLU`, and metadata-only `View` (covers `Flatten` / `Reshape`).
///
/// Reference: designs/2026-03-20-issue-4258-gpu-resident-pgd-dag-ibp-forward.md
#[derive(Clone)]
#[allow(clippy::large_enum_variant)] // Conv2d variant is larger but Boxing adds indirection on GPU dispatch hot path
pub enum GpuIbpLayer {
    /// Linear: output_lower = W+ @ input_lower + W- @ input_upper + bias
    ///         output_upper = W+ @ input_upper + W- @ input_lower + bias
    Linear {
        /// Weight matrix (out_features × in_features) row-major.
        /// Uses `Arc<[f32]>` for zero-copy sharing with CROWN cache.
        weight: Arc<[f32]>,
        /// Optional bias (out_features,)
        bias: Option<Arc<[f32]>>,
        out_features: usize,
        in_features: usize,
    },
    /// Conv2d interval propagation on GPU for groups=1.
    ///
    /// Weight uses row-major `(out_channels, in_channels, kernel_h, kernel_w)`
    /// layout so the resident shader can index it directly without reshaping.
    Conv2d {
        weight: Arc<[f32]>,
        bias: Option<Arc<[f32]>>,
        out_channels: usize,
        in_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        groups: usize,
        input_h: usize,
        input_w: usize,
    },
    /// ReLU: lower = max(lower, 0), upper = max(upper, 0)
    ReLU { num_elements: usize },
    /// Metadata-only reshape (Flatten, Reshape). No GPU buffer change —
    /// only updates the logical shape for subsequent layers.
    View { output_shape: Arc<[usize]> },
}

/// GPU-resident IBP forward pass that keeps bounds on device across all layers.
///
/// Unlike per-layer `GemmEngine` calls (which upload/download per operation),
/// this trait uploads input bounds once, encodes all layer passes into one
/// command buffer, and reads back only the final output bounds. This eliminates
/// N-1 host roundtrips for an N-layer resident chain.
///
/// Reference: designs/2026-03-18-issue-4081-gpu-ibp-forward-gap2-addendum.md
/// Part of #4081.
pub trait GpuIbpForward: Sync + Send {
    /// Run complete IBP forward pass on GPU.
    ///
    /// - `layers`: Layer descriptors in forward order (input-to-output)
    /// - `input_lower`: Flattened input lower bounds
    /// - `input_upper`: Flattened input upper bounds
    /// - `input_shape`: Input tensor shape
    ///
    /// Returns output bounds and shape after all layers.
    fn ibp_forward_gpu(
        &self,
        layers: &[GpuIbpLayer],
        input_lower: &[f32],
        input_upper: &[f32],
        input_shape: &[usize],
    ) -> Result<GpuIbpResult>;

    /// SOUND (verdict-legal) GPU IBP forward pass.
    ///
    /// Same contract as [`ibp_forward_gpu`](Self::ibp_forward_gpu), but every
    /// endpoint is a CERTIFIED enclosure: the reduction rounding is over-bounded
    /// by a directed `γ_k·S` widening, the underflow floor is NORMAL-range (Metal
    /// FTZ-safe), the weight-amplified subnormal-flush loss is covered on-device,
    /// and the outward store uses `center ∓ positive radius` — so the returned
    /// interval is a SUPERSET of both the true forward range AND the CPU
    /// `propagate_ibp_sound` bound. All arithmetic is f32 (Metal-legal).
    ///
    /// This is the IBP counterpart of
    /// [`GpuCrownBackward::crown_backward_gpu_sound`](crate::GpuCrownBackward::crown_backward_gpu_sound):
    /// usable to decide a `Verified`/`unsat`/`hold` even under the soundness gate.
    ///
    /// Default: unsupported → the caller takes the proven-sound CPU IBP path.
    fn ibp_forward_gpu_sound(
        &self,
        _layers: &[GpuIbpLayer],
        _input_lower: &[f32],
        _input_upper: &[f32],
        _input_shape: &[usize],
    ) -> Result<GpuIbpResult> {
        Err(NyError::UnsupportedOp(
            "sound GPU IBP forward not supported by this engine".into(),
        ))
    }

    /// Whether this engine advertises a verdict-legal sound GPU IBP forward
    /// (`ibp_forward_gpu_sound`). Lets the soundness gate route a verdict-deciding
    /// IBP bound onto the sound GPU path instead of the CPU fallback. Default
    /// `false`.
    fn provides_sound_gpu_ibp(&self) -> bool {
        false
    }
}

/// Cached GPU execution plan for resident IBP forward passes.
///
/// Implementations pre-upload static layer data such as weights and biases, so
/// repeated calls only need fresh input bounds and the final output readback.
pub trait GpuIbpModelPlan: Sync + Send {
    /// Run one resident IBP forward pass using cached static buffers.
    fn ibp_forward_cached(
        &self,
        input_lower: &[f32],
        input_upper: &[f32],
        input_shape: &[usize],
    ) -> Result<GpuIbpResult>;
}

/// Optional cached-plan preparation for GPU-resident IBP forward backends.
///
/// Some engines can lower a fixed network/input-shape pair into a reusable
/// execution plan. Callers should fall back to [`GpuIbpForward`] or CPU paths
/// when this returns `Ok(None)`.
pub trait GpuIbpForwardExt: GpuIbpForward {
    /// Prepare a reusable resident-IBP model plan for the given layers and
    /// input shape.
    fn prepare_model_plan(
        &self,
        layers: &[GpuIbpLayer],
        input_shape: &[usize],
    ) -> Result<Option<Box<dyn GpuIbpModelPlan>>>;
}
