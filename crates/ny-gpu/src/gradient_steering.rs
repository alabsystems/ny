// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #alpha-steering-proposal: live WGPU joint-α adjoint for gradient
//! PROPOSALS only.
//!
//! Sibling of [`crate::attack_steering::AttackSteeringDevice`] (b030e2a8) for
//! the other proposal-grade consumer class: the DAG α-CROWN margin-gradient
//! lane. When no qualified resident proof engine is present, every armed
//! margin-gradient iteration used to fall back to the single-layer LOCAL
//! gradient rule instead of the true `∂(binding-row lower bound)/∂α` adjoint —
//! wrong-direction steering that makes the ascent patience-exit, not crawl.
//!
//! Doctrine (design I3, consult #4): "Gradients may be approximate and used
//! only to propose alpha." Every α proposal is re-evaluated by the certified
//! CPU fold on the next iteration, and best-state retention rejects
//! regressions, so a proposal-grade adjoint engine is sound BY CONSTRUCTION
//! for this consumer — and for NO other consumer.
//!
//! CONTRACT (routing): this handle travels ONLY through the dedicated
//! α-gradient steering channel
//! (`ny_propagate::alpha_gradient_steering`) into the margin-gradient
//! proposal seam. It must never reach a bound / precheck / BaB-bounding call
//! site; those stay on the quarantined proof adapter. Two mechanical
//! backstops enforce that even a misrouted handle stays verdict-neutral:
//!
//! 1. `gemm_f32` REFUSES: unlike the attack channel, the α-steering channel
//!    exposes no general GEMM surface at all, so the handle is unusable as an
//!    ordinary engine.
//! 2. `as_gpu_crown_backward()` hands out this proposal wrapper, not the raw
//!    [`WgpuDevice`]. The wrapper delegates only the two joint-α gradient
//!    methods and keeps every verdict-bearing CROWN method at its refusing
//!    default, including `provides_sound_gpu_crown() == false`.
//!
//! The one live capability is the deadline-bounded joint-α adjoint
//! (`crown_joint_alpha_gradient_resident_with_deadline`), whose return type
//! is gradients only (`Vec<Vec<f32>>`) — the proposal path cannot leak a
//! bound because the API it consumes never produces one.
//!
//! The wrapper holds an `Arc` to the same ordinary WGPU context used by attack
//! and forward-linear value proposals. Sharing changes only resource ownership;
//! this wrapper's capability mask remains the authority boundary.

use crate::attack_steering::shared_proposal_wgpu_device;
use crate::wgpu_device::WgpuDevice;
use ny_core::{GemmEngine, GpuCrownBackward, NyError, Result};
use std::sync::Arc;

/// α-gradient-steering WGPU engine: exposes ONLY the joint-α adjoint proposal
/// hook. See the module docs for the soundness argument and the routing
/// contract. Constructing one NEVER weakens the proof-adapter quarantine.
pub struct GradientSteeringDevice {
    device: Arc<WgpuDevice>,
}

impl GradientSteeringDevice {
    /// Create the α-steering accelerator on the WGPU adapter.
    ///
    /// Fails (and the caller keeps the bounded local-gradient fallback) when
    /// no adapter is available; failure must never fail the verification run.
    pub fn new_wgpu() -> Result<Self> {
        Ok(Self {
            device: shared_proposal_wgpu_device()?,
        })
    }
}

impl GemmEngine for GradientSteeringDevice {
    fn backend_provenance(&self) -> &'static str {
        // Truthful identity (#backend-provenance): distinguishes the α
        // proposal handle from the CPU engine, the quarantined seam, AND the
        // attack-steering channel.
        "wgpu-alpha-steering"
    }

    /// The α-steering channel is NOT a GEMM engine. Refusing here keeps the
    /// handle unusable at any engine-shaped call site it might be misrouted
    /// to; its single consumer only ever takes `as_gpu_crown_backward()`.
    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> Result<Vec<f32>> {
        Err(NyError::UnsupportedOp(
            "wgpu-alpha-steering exposes only the joint-α adjoint proposal hook; \
             it is not a GEMM engine"
                .into(),
        ))
    }

    /// Live: the margin-gradient proposal seam drives
    /// `crown_joint_alpha_gradient_resident_with_deadline` through this hook.
    /// The returned trait object is this proposal wrapper. It delegates only
    /// joint-α gradients; every verdict-bearing method remains unavailable.
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}

impl GpuCrownBackward for GradientSteeringDevice {
    fn clear_crown_working_set(&self) -> Result<()> {
        self.device.clear_crown_working_set()
    }

    fn crown_backward_gpu(
        &self,
        _layers: &[ny_core::GpuCrownLayer],
        _spec: &[f32],
        _num_specs: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> Result<ny_core::GpuCrownResult> {
        Err(NyError::UnsupportedOp(
            "wgpu-alpha-steering exposes gradients only, never verdict-bearing CROWN".into(),
        ))
    }

    fn crown_joint_alpha_gradient_resident(
        &self,
        segments: &[ny_core::GpuResnetSegment],
        seed_lower_a: &[f32],
        num_specs: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        GpuCrownBackward::crown_joint_alpha_gradient_resident(
            self.device.as_ref(),
            segments,
            seed_lower_a,
            num_specs,
            output_dim,
            input_lower,
            input_upper,
        )
    }

    fn provides_deadline_bounded_joint_alpha_gradient_resident(&self) -> bool {
        GpuCrownBackward::provides_deadline_bounded_joint_alpha_gradient_resident(
            self.device.as_ref(),
        )
    }

    fn crown_joint_alpha_gradient_resident_with_deadline(
        &self,
        segments: &[ny_core::GpuResnetSegment],
        seed_lower_a: &[f32],
        num_specs: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
        deadline: std::time::Instant,
    ) -> Result<Vec<Vec<f32>>> {
        GpuCrownBackward::crown_joint_alpha_gradient_resident_with_deadline(
            self.device.as_ref(),
            segments,
            seed_lower_a,
            num_specs,
            output_dim,
            input_lower,
            input_upper,
            deadline,
        )
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use super::*;
    use crate::backend::{Backend, ComputeDevice};

    /// #alpha-steering-proposal quarantine pin (twin of
    /// `attack_lanes_arm_on_steering_engine_while_proof_seam_stays_closed`).
    ///
    /// Fixed side: the steering handle arms the deadline-bounded joint-α
    /// adjoint the margin-gradient proposal seam gates on. Fail-closed side:
    /// the proof adapter still refuses every verdict-bearing WGPU route, and
    /// even the steering handle's own backward device keeps
    /// `provides_sound_gpu_crown() == false` — adding this channel provably
    /// does not weaken verdict quarantine.
    #[test]
    fn alpha_steering_arms_joint_adjoint_while_proof_seam_stays_closed() {
        let steering = GradientSteeringDevice::new_wgpu().expect("adapter probed available");
        let engine: &dyn GemmEngine = &steering;
        let gpu = engine
            .as_gpu_crown_backward()
            .expect("proposal adjoint hook must be armable");
        assert!(
            gpu.provides_deadline_bounded_joint_alpha_gradient_resident(),
            "the margin-gradient proposal seam requires the deadline-bounded joint adjoint"
        );
        assert!(
            !gpu.provides_sound_gpu_crown(),
            "the steering handle must never carry verdict authority"
        );
        assert!(matches!(
            gpu.crown_backward_gpu(&[], &[], 0, &[], &[]),
            Err(NyError::UnsupportedOp(_))
        ));
        assert!(matches!(
            gpu.crown_backward_gpu_sound(&[], &[], 0, &[], &[]),
            Err(NyError::UnsupportedOp(_))
        ));
        assert!(
            engine.gemm_f32(1, 1, 1, &[1.0], &[1.0]).is_err(),
            "the α-steering channel must refuse general GEMM traffic"
        );

        // Fail-closed side, unchanged: the proof adapter must still refuse
        // every verdict-bearing WGPU route.
        let proof = ComputeDevice::new(Backend::Wgpu).expect("adapter probed available");
        let proof_engine: &dyn GemmEngine = &proof;
        assert!(proof_engine.as_gpu_crown_backward().is_none());
        assert!(proof_engine
            .gemm_f32(2, 2, 2, &[1.0, 0.0, 0.0, 1.0], &[3.0, 4.0, 5.0, 6.0])
            .is_err());
    }

    /// The steering handle must be truthfully identified in telemetry: never
    /// mistakable for the CPU engine, the quarantined seam, or the attack
    /// channel.
    #[test]
    fn alpha_steering_provenance_is_distinct() {
        let steering = GradientSteeringDevice::new_wgpu().expect("adapter probed available");
        assert_eq!(
            GemmEngine::backend_provenance(&steering),
            "wgpu-alpha-steering"
        );
        assert_ne!(
            GemmEngine::backend_provenance(&steering),
            "wgpu-attack-steering"
        );
    }
}
