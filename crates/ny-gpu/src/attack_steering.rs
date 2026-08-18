// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #attack-steering-unquarantine: live WGPU acceleration for falsification
//! STEERING only.
//!
//! An ordinary [`ComputeDevice`](crate::ComputeDevice) / [`WgpuDevice`] remains
//! unqualified for proof use. The explicit proof constructor can open only its
//! exact device's reviewed CROWN seam after live qualification; none of that
//! authority is inherited by this proposal wrapper.
//!
//! Applying it to the sat-finding attack lanes, however, over-reached: the
//! quarantine left the CLI with `gemm_engine = None`, which STATICALLY
//! disarms the batched and exact-VJP disjunctive PGD attacks
//! (`gemm_engine.is_some()` gates in `graph_pgd.rs`) and degrades candidate
//! search to the slow sequential loop. Measured on cifar100_2024 medium rows
//! 1592/4752 (VNN-COMP 100 s budget): the engine-side attack that found the
//! near-boundary counterexample in <1 s on the accelerator can no longer
//! converge inside its slice, and genuine fast `sat` rows became timeouts.
//!
//! Attack steering is verdict-neutral BY CONSTRUCTION — the float results of
//! this engine only choose WHERE to look next. A candidate becomes `sat`
//! exclusively through the unchanged admission gates: the engine-side
//! `re_evaluate_and_confirm` (noise-scaled margin + per-clause box + global
//! attack-box guard, #witness-box-audit) and, on the scored VNN-COMP path,
//! the trusted-ORT + true-f64 gate (`SatGated`,
//! docs/VERDICT_ADMISSION_SPEC.md). A wrong float here can only waste attack
//! budget; `unsat` bounds NEVER flow through this handle. This is the same
//! consumer class `GemmEngine::gemm_f32_fast` documents as "ONLY for
//! soundness-free consumers — adversarial attack / counterexample search".
//!
//! CONTRACT (routing, enforced by the CLI): pass this handle ONLY through the
//! dedicated `attack_gemm_engine` channel to falsification call sites. It
//! must never reach a bound / precheck / BaB-bounding call site; those stay
//! on the explicitly qualified proof adapter.
//!
//! # Shared ordinary context
//!
//! Attack, alpha-gradient, and forward-linear value proposals are capability
//! views over one process-global ordinary [`WgpuDevice`]. They must not each
//! create an adapter/device/pipeline context: doing so multiplies the resident
//! driver and pipeline footprint and can OOM before verification starts. The
//! shared raw device is deliberately constructed with [`WgpuDevice::new`], so
//! it has no proof report; each wrapper below still exposes only its own narrow
//! proposal surface.

use crate::wgpu_device::WgpuDevice;
use ny_core::{ConvTranspose2dParams, GemmEngine, GpuCrownBackward, NyError, Result};
use std::sync::{Arc, OnceLock};

type SharedContextSlot<T> = OnceLock<std::result::Result<Arc<T>, Arc<str>>>;

/// Resolve one fallible process-shared context without requiring its error type
/// to be cloneable. A failed initialization is cached as text so later proposal
/// seams do not race a second device construction after the first one refused.
fn shared_context_with<T, F>(slot: &SharedContextSlot<T>, initialize: F) -> Result<Arc<T>>
where
    F: FnOnce() -> Result<T>,
{
    match slot.get_or_init(|| {
        initialize()
            .map(Arc::new)
            .map_err(|error| Arc::<str>::from(error.to_string()))
    }) {
        Ok(context) => Ok(Arc::clone(context)),
        Err(reason) => Err(NyError::UnsupportedConfiguration(format!(
            "shared ordinary WGPU proposal context unavailable: {reason}"
        ))),
    }
}

/// One ordinary, unqualified WGPU context shared by every proposal wrapper.
///
/// This is crate-private so external callers cannot confuse context ownership
/// with a new capability. Authority continues to be defined by the wrapper's
/// trait implementation, not by possession of this `Arc`.
pub(crate) fn shared_proposal_wgpu_device() -> Result<Arc<WgpuDevice>> {
    static DEVICE: SharedContextSlot<WgpuDevice> = OnceLock::new();
    shared_context_with(&DEVICE, WgpuDevice::new)
}

/// Attack-only WGPU engine: exposes the live WGPU kernels for falsification
/// steering. See the module docs for the soundness argument and the routing
/// contract. Constructing one NEVER weakens the proof-adapter quarantine.
pub struct AttackSteeringDevice {
    device: Arc<WgpuDevice>,
}

impl AttackSteeringDevice {
    /// Create the attack-steering accelerator on the WGPU adapter.
    ///
    /// Fails (and the caller falls back to CPU steering) when no adapter is
    /// available; failure must never fail the verification run itself.
    pub fn new_wgpu() -> Result<Self> {
        Ok(Self {
            device: shared_proposal_wgpu_device()?,
        })
    }
}

impl GemmEngine for AttackSteeringDevice {
    fn backend_provenance(&self) -> &'static str {
        // Truthful identity (#backend-provenance): distinguishes the attack
        // steering handle from both the CPU engine and the quarantined seam.
        "wgpu-attack-steering"
    }

    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.device.gemm_f32(m, k, n, a, b)
    }

    // `gemm_f32_fast` keeps the trait default (routes to `gemm_f32` above):
    // the WGSL kernel is already plain RN-f32.

    fn conv_transpose_2d(
        &self,
        a_reshaped: &[f32],
        weight_col: &[f32],
        params: &ConvTranspose2dParams,
    ) -> Result<Vec<f32>> {
        self.device
            .conv_transpose_2d(a_reshaped, weight_col, params)
    }

    fn conv_transpose_2d_pair_cached(
        &self,
        a_lower: &[f32],
        a_upper: &[f32],
        weight_col: &Arc<[f32]>,
        params: &ConvTranspose2dParams,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.device
            .conv_transpose_2d_pair_cached(a_lower, a_upper, weight_col, params)
    }

    /// Live: the exact-VJP batched attack (`graph_pgd_vjp_batched`) drives its
    /// ATTACK GRADIENTS through this hook. The returned trait object is this
    /// proposal wrapper, which delegates only point-VJP methods and keeps all
    /// verdict-bearing CROWN methods unavailable.
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }

    /// Live: concrete-point forwards (`propagate_concrete_point`) ride the
    /// cached GPU DAG plan through these hooks; a point box is a network
    /// VALUE, not a bound.
    fn as_gpu_ibp_forward(&self) -> Option<&dyn ny_core::GpuIbpForward> {
        Some(self.device.as_ref())
    }

    fn as_gpu_ibp_forward_ext(&self) -> Option<&dyn ny_core::GpuIbpForwardExt> {
        Some(self.device.as_ref())
    }

    fn as_gpu_dag_ibp_forward_ext(&self) -> Option<&dyn ny_core::GpuDagIbpForwardExt> {
        Some(self.device.as_ref())
    }
}

impl GpuCrownBackward for AttackSteeringDevice {
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
            "wgpu-attack-steering exposes point-VJP only, never verdict-bearing CROWN".into(),
        ))
    }

    fn crown_point_vjp_batched(
        &self,
        layers_backward: &[ny_core::GpuCrownLayer],
        mask_positions: &[usize],
        masks: &[Vec<Vec<f32>>],
        spec_rows: &[f32],
        output_dim: usize,
        input_dim: usize,
    ) -> Result<Vec<Vec<f32>>> {
        GpuCrownBackward::crown_point_vjp_batched(
            self.device.as_ref(),
            layers_backward,
            mask_positions,
            masks,
            spec_rows,
            output_dim,
            input_dim,
        )
    }

    fn crown_point_vjp_batched_resnet(
        &self,
        segments_backward: &[ny_core::GpuResnetSegment],
        mask_flat_positions: &[usize],
        masks: &[Vec<Vec<f32>>],
        spec_rows: &[f32],
        output_dim: usize,
        input_dim: usize,
    ) -> Result<Vec<Vec<f32>>> {
        GpuCrownBackward::crown_point_vjp_batched_resnet(
            self.device.as_ref(),
            segments_backward,
            mask_flat_positions,
            masks,
            spec_rows,
            output_dim,
            input_dim,
        )
    }
}

#[cfg(test)]
mod shared_context_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;

    #[derive(Debug)]
    struct HermeticContext;

    /// No adapter is needed: this pins the exact construction/identity property
    /// used by all three real proposal wrappers.
    #[test]
    fn shared_context_constructs_once_and_returns_pointer_identical_arcs() {
        const CONTENDERS: usize = 8;
        let slot = Arc::new(SharedContextSlot::<HermeticContext>::new());
        let constructions = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(CONTENDERS));

        let handles: Vec<_> = (0..CONTENDERS)
            .map(|_| {
                let slot = Arc::clone(&slot);
                let constructions = Arc::clone(&constructions);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    shared_context_with(slot.as_ref(), || {
                        constructions.fetch_add(1, Ordering::SeqCst);
                        std::thread::yield_now();
                        Ok(HermeticContext)
                    })
                    .expect("shared context construction")
                })
            })
            .collect();
        let contexts: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().expect("shared-context contender panicked"))
            .collect();

        assert_eq!(constructions.load(Ordering::SeqCst), 1);
        assert!(contexts
            .iter()
            .skip(1)
            .all(|context| Arc::ptr_eq(&contexts[0], context)));
    }

    /// A refused adapter/device initialization is final for the process-wide
    /// slot too. Retrying from a sibling wrapper would violate the one-context
    /// contract and can repeat a driver's largest transient allocation.
    #[test]
    fn shared_context_caches_failure_without_retrying_construction() {
        let slot = SharedContextSlot::<HermeticContext>::new();
        let constructions = AtomicUsize::new(0);

        let first = shared_context_with(&slot, || {
            constructions.fetch_add(1, Ordering::SeqCst);
            Err(NyError::UnsupportedConfiguration(
                "hermetic adapter refusal".into(),
            ))
        })
        .expect_err("first construction must preserve the refusal");
        let second = shared_context_with(&slot, || {
            constructions.fetch_add(1, Ordering::SeqCst);
            panic!("a refused shared-context slot must not retry construction")
        })
        .expect_err("second lookup must return the cached refusal");

        assert_eq!(constructions.load(Ordering::SeqCst), 1);
        assert!(first.to_string().contains("hermetic adapter refusal"));
        assert!(second.to_string().contains("hermetic adapter refusal"));
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use super::*;
    use crate::backend::{Backend, ComputeDevice};

    /// #attack-steering-unquarantine seam pin, regression side.
    ///
    /// This is the exact rejection condition that killed the cifar100 medium
    /// fast sats (rows 1592/4752) after the 1ede1d30 quarantine: the ONLY
    /// engine channel was the quarantined proof adapter, so the batched /
    /// exact-VJP attack lanes (`gemm_engine.is_some()` +
    /// `as_gpu_crown_backward()` gates in `graph_pgd*.rs`) could never arm.
    /// Under 1ede1d30's logic there is no constructible engine for which BOTH
    /// halves below hold, so this test FAILS there and passes with the
    /// attack-steering channel.
    #[test]
    fn attack_lanes_arm_on_steering_engine_while_proof_seam_stays_closed() {
        // Fixed side: the attack channel exposes the live steering routes the
        // PGD lanes gate on.
        let steering = AttackSteeringDevice::new_wgpu().expect("adapter probed available");
        let engine: &dyn GemmEngine = &steering;
        let gpu = engine
            .as_gpu_crown_backward()
            .expect("exact-VJP batched attack lane must be armable (attack gradients)");
        assert!(
            !gpu.provides_sound_gpu_crown(),
            "the attack-steering handle must never carry verdict authority"
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
            engine.as_gpu_dag_ibp_forward_ext().is_some(),
            "cached DAG point-forward must be armable (attack candidate scoring)"
        );
        let c = engine
            .gemm_f32(2, 2, 2, &[1.0, 0.0, 0.0, 1.0], &[3.0, 4.0, 5.0, 6.0])
            .expect("attack-steering GEMM is live");
        assert_eq!(c, vec![3.0, 4.0, 5.0, 6.0]);

        // Fail-closed side, unchanged: the proof adapter must still refuse
        // every verdict-bearing WGPU route.
        let proof = ComputeDevice::new(Backend::Wgpu).expect("adapter probed available");
        let proof_engine: &dyn GemmEngine = &proof;
        assert!(
            proof_engine
                .gemm_f32(2, 2, 2, &[1.0, 0.0, 0.0, 1.0], &[3.0, 4.0, 5.0, 6.0])
                .is_err(),
            "proof-adapter WGPU GEMM must stay quarantined"
        );
        assert!(proof_engine.as_gpu_crown_backward().is_none());
        assert!(proof_engine.as_gpu_dag_ibp_forward_ext().is_none());
    }

    /// The steering handle must be truthfully identified in telemetry: never
    /// mistakable for the CPU engine or the quarantined seam.
    #[test]
    fn steering_provenance_is_distinct() {
        let steering = AttackSteeringDevice::new_wgpu().expect("adapter probed available");
        assert_eq!(
            GemmEngine::backend_provenance(&steering),
            "wgpu-attack-steering"
        );
    }
}
