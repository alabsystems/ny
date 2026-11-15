// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

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

    /// A cached plan may opt into execution with the caller's original absolute
    /// deadline. Legacy implementations are never called through this entry.
    fn ibp_forward_cached_with_deadline(
        &self,
        _input_lower: &[f32],
        _input_upper: &[f32],
        _input_shape: &[usize],
        deadline: Instant,
    ) -> Result<GpuIbpResult> {
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "cached GPU IBP: original deadline expired".into(),
            ));
        }
        Err(NyError::UnsupportedOp(
            "cached GPU IBP plan does not support deadline execution".into(),
        ))
    }

    /// Supporting cached execution alone does not grant deadline capability.
    fn supports_deadline_execution(&self) -> bool {
        false
    }
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

    /// Prepare using the original caller deadline; do not restart a duration or
    /// invoke legacy preparation without a deadline. None admits CPU fallback.
    fn prepare_model_plan_with_deadline(
        &self,
        _layers: &[GpuIbpLayer],
        _input_shape: &[usize],
        deadline: Instant,
    ) -> Result<Option<Box<dyn GpuIbpModelPlan>>> {
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "GPU IBP plan preparation: original deadline expired".into(),
            ));
        }
        Ok(None)
    }
}

#[cfg(test)]
mod deadline_plan_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct Legacy(AtomicUsize);
    impl GpuIbpForward for Legacy {
        fn ibp_forward_gpu(
            &self,
            _: &[GpuIbpLayer],
            _: &[f32],
            _: &[f32],
            _: &[usize],
        ) -> Result<GpuIbpResult> {
            self.0.fetch_add(1, Ordering::Relaxed);
            panic!("legacy resident call must not run during deadline preparation");
        }
    }
    impl GpuIbpForwardExt for Legacy {
        fn prepare_model_plan(
            &self,
            _: &[GpuIbpLayer],
            _: &[usize],
        ) -> Result<Option<Box<dyn GpuIbpModelPlan>>> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(None)
        }
    }
    impl GpuIbpModelPlan for Legacy {
        fn ibp_forward_cached(&self, _: &[f32], _: &[f32], _: &[usize]) -> Result<GpuIbpResult> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(GpuIbpResult {
                lower_bounds: vec![1.0],
                upper_bounds: vec![1.0],
                output_shape: vec![1],
            })
        }
    }

    #[test]
    fn finite_deadline_defaults_never_call_legacy_plan_routes() {
        let backend = Legacy(AtomicUsize::new(0));
        let live = Instant::now() + Duration::from_secs(10);
        assert!(backend
            .prepare_model_plan_with_deadline(&[], &[1], live)
            .unwrap()
            .is_none());
        assert!(!backend.supports_deadline_execution());
        assert!(matches!(
            backend.ibp_forward_cached_with_deadline(&[1.0], &[1.0], &[1], live),
            Err(NyError::UnsupportedOp(_))
        ));
        assert_eq!(backend.0.load(Ordering::Relaxed), 0);
        backend.prepare_model_plan(&[], &[1]).unwrap();
        let value = backend.ibp_forward_cached(&[1.0], &[1.0], &[1]).unwrap();
        assert_eq!(value.lower_bounds, [1.0]);
        assert_eq!(backend.0.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn expired_deadline_defaults_refuse_before_legacy_work() {
        let backend = Legacy(AtomicUsize::new(0));
        let expired = Instant::now();
        assert!(matches!(
            backend.prepare_model_plan_with_deadline(&[], &[1], expired),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert!(matches!(
            backend.ibp_forward_cached_with_deadline(&[1.0], &[1.0], &[1], expired),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert_eq!(backend.0.load(Ordering::Relaxed), 0);
    }
}
