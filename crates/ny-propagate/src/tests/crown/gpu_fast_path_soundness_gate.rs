// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Routing test for the GPU CROWN soundness gate (#vnncomp-gpu-crown-soundness).
//!
//! VNN-COMP scores one incorrect verdict at -150. The GPU CROWN backward /
//! concretize path is f32 round-to-nearest with NO γ_n·S certified error, so a
//! GPU-derived bound can be *tighter than the true range* — exactly the kind of
//! over-tight bound that flips a genuinely-violated instance to Verified/unsat.
//!
//! These tests prove the gate routes the *verdict-deciding* CROWN bound onto the
//! proven-sound CPU path when soundness is required, even though a fully GPU-CROWN
//! -capable engine is provided and the model is GPU-eligible. Because this dev box
//! has no usable GPU, we assert by ROUTING: a scripted GPU engine returns a
//! deliberately-poisoned (over-tight, beyond IBP) bound and counts its own calls.
//!
//! - gate OFF (speed-only): the GPU engine IS consulted and the poisoned bound
//!   flows through — confirming the model is genuinely GPU-eligible and the engine
//!   is wired into the verdict path. (This is the unsound behavior the gate fixes.)
//! - gate ON  (soundness required): the GPU engine is NOT consulted for the
//!   verdict bound; the result equals the CPU sound bound, NOT the poisoned GPU
//!   bound.

use super::*;
use crate::sound_gpu_gate::{is_sound_gpu_crown_required, set_sound_gpu_crown_required};
use ndarray::{arr1, arr2};
use ny_core::{
    GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownResult, NaiveCpuGemmEngine, Result,
};
use std::sync::atomic::{AtomicUsize, Ordering};

/// The gate is a process-global; use the ONE shared lock (`sound_gpu_gate::
/// test_lock`) so gate-flipping tests here exclude gate-dependent tests in
/// other modules too, and the default (disabled) is restored on exit.
use crate::sound_gpu_gate::test_lock::lock_gate;

/// A GPU CROWN engine that returns a deliberately-POISONED bound and counts how
/// many times the verdict path actually dispatched to it. If the gate ever lets
/// this engine decide the verdict bound, the result is provably unsound.
struct PoisonedGpuCrownEngine {
    lower_bounds: Vec<f32>,
    upper_bounds: Vec<f32>,
    gpu_calls: AtomicUsize,
}

impl PoisonedGpuCrownEngine {
    fn new(lower_bounds: Vec<f32>, upper_bounds: Vec<f32>) -> Self {
        Self {
            lower_bounds,
            upper_bounds,
            gpu_calls: AtomicUsize::new(0),
        }
    }
    fn gpu_calls(&self) -> usize {
        self.gpu_calls.load(Ordering::SeqCst)
    }
}

impl GemmEngine for PoisonedGpuCrownEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        // Delegate the (sound) GEMM to the CPU engine so the CPU CROWN fallback
        // still works when the gate masks our GPU CROWN backward.
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }

    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}

impl GpuCrownBackward for PoisonedGpuCrownEngine {
    fn crown_backward_gpu(
        &self,
        _layers: &[GpuCrownLayer],
        _spec: &[f32],
        num_specs: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        self.gpu_calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(
            self.lower_bounds.len(),
            num_specs,
            "poisoned GPU engine expects one scalar bound per output spec"
        );
        Ok(GpuCrownResult {
            lower_bounds: self.lower_bounds.clone(),
            upper_bounds: self.upper_bounds.clone(),
        })
    }
}

fn build_linear_relu_network() -> Result<(Network, BoundedTensor)> {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.4, -0.1], [0.2, 0.3], [-0.5, 0.7]]),
        Some(arr1(&[0.1, -0.2, 0.05])),
    )?));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]),
        Some(arr1(&[0.0, 0.15])),
    )?));

    let input = BoundedTensor::new(
        arr1(&[-1.0, -0.5]).into_dyn(),
        arr1(&[1.0, 0.75]).into_dyn(),
    )?;
    Ok((network, input))
}

/// Build a poisoned GPU bound that is strictly TIGHTER than the CPU sound bound
/// (and tighter than IBP) — i.e. an unsound over-claim. We shrink the interval
/// toward its midpoint so it could not contain the true output range.
fn poisoned_tighter_than_cpu(cpu: &BoundedTensor) -> (Vec<f32>, Vec<f32>) {
    let lower: Vec<f32> = cpu.lower().iter().copied().collect();
    let upper: Vec<f32> = cpu.upper().iter().copied().collect();
    let mut p_lower = lower.clone();
    let mut p_upper = upper.clone();
    for i in 0..lower.len() {
        let mid = f32::midpoint(lower[i], upper[i]);
        // Pull both endpoints 90% of the way to the midpoint: a much-too-tight box.
        p_lower[i] = lower[i] + 0.9 * (mid - lower[i]);
        p_upper[i] = upper[i] - 0.9 * (upper[i] - mid);
    }
    (p_lower, p_upper)
}

/// Run the verdict-deciding CROWN backward over caller-supplied (sound) layer
/// bounds — this is the path the GPU final-concretize fast-path lives on. Using
/// the layer-bounds entry (like the nan-guard test) isolates the verdict
/// concretize from the IBP-collection pass, so the scripted engine sees exactly
/// one dispatch with `num_specs == output_dim`.
fn crown_verdict_with_engine(
    network: &Network,
    input: &BoundedTensor,
    layer_bounds: &[BoundedTensor],
    engine: &dyn GemmEngine,
) -> Result<BoundedTensor> {
    network.propagate_crown_with_layer_bounds_and_engine_and_deadline_and_limits(
        input,
        layer_bounds,
        Some(engine),
        None,
        None,
    )
}

/// Sanity: with the gate OFF (speed-only path), the GPU engine IS consulted for
/// the verdict bound and its poisoned (over-tight) result flows through. This
/// confirms the model is GPU-eligible and the engine is genuinely on the verdict
/// path — so the gate-ON assertion below is meaningful, not vacuous.
#[test]
fn gate_off_gpu_poisoned_bound_reaches_verdict() -> Result<()> {
    let _g = lock_gate();
    let (network, input) = build_linear_relu_network()?;
    assert!(
        network.is_gpu_crown_eligible(),
        "fixture must be GPU-CROWN-eligible so the GPU path is reachable"
    );

    let cpu_engine = NaiveCpuGemmEngine;
    let layer_bounds = network.collect_crown_ibp_bounds_with_engine_and_deadline(
        &input,
        Some(&cpu_engine),
        None,
    )?;
    let cpu_sound = crown_verdict_with_engine(&network, &input, &layer_bounds, &cpu_engine)?;
    let (p_lower, p_upper) = poisoned_tighter_than_cpu(&cpu_sound);
    let gpu = PoisonedGpuCrownEngine::new(p_lower.clone(), p_upper.clone());

    // Gate OFF.
    assert!(!is_sound_gpu_crown_required());
    let out = crown_verdict_with_engine(&network, &input, &layer_bounds, &gpu)?;

    assert_eq!(
        gpu.gpu_calls(),
        1,
        "with the gate OFF the verdict path must dispatch to the GPU engine"
    );
    // The GPU result is intersected with the (sound) forward bounds in
    // tighten_crown_output. Since the poisoned bounds are tighter than CPU/IBP,
    // the intersection keeps the poisoned (over-tight) endpoints — i.e. the
    // verdict would be decided on the GPU's unsound bound. We assert it is at
    // least as tight as the poisoned bound on every endpoint.
    for i in 0..out.len() {
        assert!(
            out.lower().as_slice().unwrap()[i] >= p_lower[i] - 1e-5,
            "gate-off lower[{i}] should be >= poisoned (over-tight) GPU lower"
        );
        assert!(
            out.upper().as_slice().unwrap()[i] <= p_upper[i] + 1e-5,
            "gate-off upper[{i}] should be <= poisoned (over-tight) GPU upper"
        );
    }
    Ok(())
}

/// THE GATE. With soundness required, the verdict bound must come from the
/// proven-sound CPU path — NOT the GPU concretize — even though a GPU-CROWN
/// -capable engine is provided and the model is GPU-eligible.
#[test]
fn gate_on_verdict_bound_is_cpu_sound_not_gpu() -> Result<()> {
    let _g = lock_gate();
    let (network, input) = build_linear_relu_network()?;
    assert!(
        network.is_gpu_crown_eligible(),
        "fixture must be GPU-CROWN-eligible so the gate has something to bypass"
    );

    let cpu_engine = NaiveCpuGemmEngine;
    let layer_bounds = network.collect_crown_ibp_bounds_with_engine_and_deadline(
        &input,
        Some(&cpu_engine),
        None,
    )?;
    let cpu_sound = crown_verdict_with_engine(&network, &input, &layer_bounds, &cpu_engine)?;
    let (p_lower, p_upper) = poisoned_tighter_than_cpu(&cpu_sound);
    let gpu = PoisonedGpuCrownEngine::new(p_lower.clone(), p_upper.clone());

    // Engage the soundness gate, then run with the GPU-capable engine.
    set_sound_gpu_crown_required(true);
    assert!(is_sound_gpu_crown_required());
    let out = crown_verdict_with_engine(&network, &input, &layer_bounds, &gpu)?;

    // 1) Routing: the GPU CROWN backward was NEVER dispatched for the verdict.
    assert_eq!(
        gpu.gpu_calls(),
        0,
        "with soundness required, the verdict path must NOT dispatch to the GPU \
         CROWN backward — it must take the proven-sound CPU route"
    );

    // 2) Result: the verdict bound equals the CPU sound bound, NOT the poisoned
    //    (over-tight) GPU bound. The gated run must be sound (>= true range), so
    //    it must be at least as WIDE as the poisoned box on at least one endpoint.
    let cpu_l = cpu_sound.lower().as_slice().unwrap().to_vec();
    let cpu_u = cpu_sound.upper().as_slice().unwrap().to_vec();
    let out_l = out.lower().as_slice().unwrap().to_vec();
    let out_u = out.upper().as_slice().unwrap().to_vec();
    for i in 0..out.len() {
        assert!(
            (out_l[i] - cpu_l[i]).abs() < 1e-5,
            "gated lower[{i}] must match CPU sound ({}) not poisoned GPU ({}); got {}",
            cpu_l[i],
            p_lower[i],
            out_l[i]
        );
        assert!(
            (out_u[i] - cpu_u[i]).abs() < 1e-5,
            "gated upper[{i}] must match CPU sound ({}) not poisoned GPU ({}); got {}",
            cpu_u[i],
            p_upper[i],
            out_u[i]
        );
    }

    // 3) Soundness sanity: the gated bound is strictly wider than the poisoned
    //    over-tight GPU bound on at least one endpoint (i.e. the gate rejected the
    //    unsound tightening on the verdict path).
    let strictly_wider =
        (0..out.len()).any(|i| out_l[i] + 1e-6 < p_lower[i] || out_u[i] - 1e-6 > p_upper[i]);
    assert!(
        strictly_wider,
        "gated CPU-sound bound must be wider than the poisoned GPU bound on at \
         least one endpoint — proving the unsound GPU tightening was rejected"
    );
    Ok(())
}

/// A GPU CROWN engine that ADVERTISES a sound backward. Its unsound
/// `crown_backward_gpu` is poisoned (must never decide a verdict under the gate),
/// while `crown_backward_gpu_sound` returns a supplied SOUND bound and counts its
/// own calls. Mirrors `WgpuDevice`, which provides a certified GPU-resident
/// sound backward. `with_deadline_support()` additionally mirrors the
/// charged-Metal/WgpuDevice cooperative-cancellation claim
/// (`honors_crown_backward_deadline == true`); without it the engine models the
/// CUDA shape, whose global deadline claim is deliberately narrow.
struct SoundGpuCrownEngine {
    sound_lower: Vec<f32>,
    sound_upper: Vec<f32>,
    poisoned_lower: Vec<f32>,
    poisoned_upper: Vec<f32>,
    unsound_calls: AtomicUsize,
    sound_calls: AtomicUsize,
    honors_deadline: bool,
    deadline_writes: std::sync::Mutex<Vec<Option<std::time::Instant>>>,
}

impl SoundGpuCrownEngine {
    fn new(
        sound_lower: Vec<f32>,
        sound_upper: Vec<f32>,
        poisoned_lower: Vec<f32>,
        poisoned_upper: Vec<f32>,
    ) -> Self {
        Self {
            sound_lower,
            sound_upper,
            poisoned_lower,
            poisoned_upper,
            unsound_calls: AtomicUsize::new(0),
            sound_calls: AtomicUsize::new(0),
            honors_deadline: false,
            deadline_writes: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn with_deadline_support(mut self) -> Self {
        self.honors_deadline = true;
        self
    }

    fn deadline_writes(&self) -> Vec<Option<std::time::Instant>> {
        self.deadline_writes
            .lock()
            .expect("deadline_writes mutex should not be poisoned")
            .clone()
    }
}

impl GemmEngine for SoundGpuCrownEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}

impl GpuCrownBackward for SoundGpuCrownEngine {
    fn crown_backward_gpu(
        &self,
        _layers: &[GpuCrownLayer],
        _spec: &[f32],
        _num_specs: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        self.unsound_calls.fetch_add(1, Ordering::SeqCst);
        Ok(GpuCrownResult {
            lower_bounds: self.poisoned_lower.clone(),
            upper_bounds: self.poisoned_upper.clone(),
        })
    }
    fn crown_backward_gpu_sound(
        &self,
        _layers: &[GpuCrownLayer],
        _spec: &[f32],
        _num_specs: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        self.sound_calls.fetch_add(1, Ordering::SeqCst);
        Ok(GpuCrownResult {
            lower_bounds: self.sound_lower.clone(),
            upper_bounds: self.sound_upper.clone(),
        })
    }
    fn provides_sound_gpu_crown(&self) -> bool {
        true
    }
    fn honors_crown_backward_deadline(&self) -> bool {
        self.honors_deadline
    }
    fn set_crown_backward_deadline(&self, deadline: Option<std::time::Instant>) {
        self.deadline_writes
            .lock()
            .expect("deadline_writes mutex should not be poisoned")
            .push(deadline);
    }
}

/// THE NEW PATH. With soundness required AND an engine that advertises a sound
/// GPU-resident backward, the verdict bound must be decided by
/// `crown_backward_gpu_sound` (certified), NOT by the unsound `crown_backward_gpu`,
/// and NOT by the slow CPU loop. This is what puts cifar100/tinyimagenet verdicts
/// on the fast sound-GPU path. The sound bound returned here equals the CPU sound
/// bound, so the verdict is unchanged — and provably sound.
#[test]
fn gate_on_sound_gpu_path_decides_verdict() -> Result<()> {
    let _g = lock_gate();
    let (network, input) = build_linear_relu_network()?;
    assert!(network.is_gpu_crown_eligible());

    let cpu_engine = NaiveCpuGemmEngine;
    let layer_bounds = network.collect_crown_ibp_bounds_with_engine_and_deadline(
        &input,
        Some(&cpu_engine),
        None,
    )?;
    let cpu_sound = crown_verdict_with_engine(&network, &input, &layer_bounds, &cpu_engine)?;
    let sound_lower: Vec<f32> = cpu_sound.lower().iter().copied().collect();
    let sound_upper: Vec<f32> = cpu_sound.upper().iter().copied().collect();
    // The unsound path is poisoned (over-tight): if it ever decides the verdict the
    // result would be wrong — so reaching the right (CPU-equal) bound proves the
    // SOUND method decided it.
    let (p_lower, p_upper) = poisoned_tighter_than_cpu(&cpu_sound);
    let gpu = SoundGpuCrownEngine::new(sound_lower.clone(), sound_upper.clone(), p_lower, p_upper);

    set_sound_gpu_crown_required(true);
    let out = crown_verdict_with_engine(&network, &input, &layer_bounds, &gpu)?;

    assert_eq!(
        gpu.sound_calls.load(Ordering::SeqCst),
        1,
        "gated verdict must dispatch to the SOUND GPU backward exactly once"
    );
    assert_eq!(
        gpu.unsound_calls.load(Ordering::SeqCst),
        0,
        "the unsound GPU backward must NEVER decide a gated verdict"
    );
    // The verdict equals the (sound) bound the GPU-sound path returned.
    for i in 0..out.len() {
        assert!(
            (out.lower().as_slice().unwrap()[i] - sound_lower[i]).abs() < 1e-4,
            "gated lower[{i}] must come from the sound GPU bound"
        );
        assert!(
            (out.upper().as_slice().unwrap()[i] - sound_upper[i]).abs() < 1e-4,
            "gated upper[{i}] must come from the sound GPU bound"
        );
    }
    Ok(())
}

/// #charged-metal-engagement (deadline-route decision). Under FINITE authority
/// the sequential GPU fast-path may now be decided by an ALREADY-MATERIALIZED
/// SOUND backend that honors cooperative cancellation — the charged-Metal /
/// `WgpuDevice` capability shape. The route consults no lazy factory under a
/// deadline (`select_lazy_backend_for_deadline`), the exact backend receives a
/// scoped cooperative lease (installed then cleared), and the verdict equals
/// the certified sound bound, never the poisoned unsound one.
#[test]
fn finite_deadline_dispatches_materialized_sound_cooperative_backend() -> Result<()> {
    let _g = lock_gate();
    let (network, input) = build_linear_relu_network()?;
    assert!(network.is_gpu_crown_eligible());

    let cpu_engine = NaiveCpuGemmEngine;
    let layer_bounds = network.collect_crown_ibp_bounds_with_engine_and_deadline(
        &input,
        Some(&cpu_engine),
        None,
    )?;
    let cpu_sound = crown_verdict_with_engine(&network, &input, &layer_bounds, &cpu_engine)?;
    let sound_lower: Vec<f32> = cpu_sound.lower().iter().copied().collect();
    let sound_upper: Vec<f32> = cpu_sound.upper().iter().copied().collect();
    let (p_lower, p_upper) = poisoned_tighter_than_cpu(&cpu_sound);
    let gpu = SoundGpuCrownEngine::new(sound_lower.clone(), sound_upper.clone(), p_lower, p_upper)
        .with_deadline_support();

    set_sound_gpu_crown_required(true);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let out = network.propagate_crown_with_layer_bounds_and_engine_and_deadline_and_limits(
        &input,
        &layer_bounds,
        Some(&gpu),
        Some(deadline),
        None,
    )?;

    assert_eq!(
        gpu.sound_calls.load(Ordering::SeqCst),
        1,
        "finite authority must dispatch the SOUND GPU backward exactly once when the \
         backend is materialized and cooperative"
    );
    assert_eq!(
        gpu.unsound_calls.load(Ordering::SeqCst),
        0,
        "the unsound GPU backward must NEVER decide a gated verdict, deadline or not"
    );
    assert_eq!(
        gpu.deadline_writes(),
        vec![Some(deadline), None],
        "the exact-backend cooperative lease must be installed for the dispatch and \
         cleared afterwards"
    );
    for i in 0..out.len() {
        assert!(
            (out.lower().as_slice().unwrap()[i] - sound_lower[i]).abs() < 1e-4,
            "deadline-admitted lower[{i}] must come from the sound GPU bound"
        );
        assert!(
            (out.upper().as_slice().unwrap()[i] - sound_upper[i]).abs() < 1e-4,
            "deadline-admitted upper[{i}] must come from the sound GPU bound"
        );
    }
    Ok(())
}

/// Byte-identity pin for the CUDA capability shape (#charged-metal-engagement):
/// a SOUND backend that leaves the cooperative-deadline capability at its
/// default (`honors_crown_backward_deadline == false` — CUDA's global claim is
/// deliberately narrow) must be refused under finite authority exactly as the
/// pre-engagement blanket skip refused it: no GPU method is consulted, no
/// device lease is installed, and the deadline-aware CPU loop decides the same
/// sound bound.
#[test]
fn finite_deadline_refuses_sound_backend_without_cooperative_cancellation() -> Result<()> {
    let _g = lock_gate();
    let (network, input) = build_linear_relu_network()?;

    let cpu_engine = NaiveCpuGemmEngine;
    let layer_bounds = network.collect_crown_ibp_bounds_with_engine_and_deadline(
        &input,
        Some(&cpu_engine),
        None,
    )?;
    let cpu_sound = crown_verdict_with_engine(&network, &input, &layer_bounds, &cpu_engine)?;
    let (p_lower, p_upper) = poisoned_tighter_than_cpu(&cpu_sound);
    let gpu = SoundGpuCrownEngine::new(
        cpu_sound.lower().iter().copied().collect(),
        cpu_sound.upper().iter().copied().collect(),
        p_lower,
        p_upper,
    );

    set_sound_gpu_crown_required(true);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let out = network.propagate_crown_with_layer_bounds_and_engine_and_deadline_and_limits(
        &input,
        &layer_bounds,
        Some(&gpu),
        Some(deadline),
        None,
    )?;

    assert_eq!(
        gpu.sound_calls.load(Ordering::SeqCst),
        0,
        "a noncooperative sound backend must not be dispatched under finite authority"
    );
    assert_eq!(gpu.unsound_calls.load(Ordering::SeqCst), 0);
    assert!(
        gpu.deadline_writes().is_empty(),
        "a refused backend must never receive a device lease"
    );
    for i in 0..out.len() {
        assert!(
            (out.lower().as_slice().unwrap()[i] - cpu_sound.lower().as_slice().unwrap()[i]).abs()
                < 1e-5,
            "refusal must leave the deadline-aware CPU sound bound deciding lower[{i}]"
        );
        assert!(
            (out.upper().as_slice().unwrap()[i] - cpu_sound.upper().as_slice().unwrap()[i]).abs()
                < 1e-5,
            "refusal must leave the deadline-aware CPU sound bound deciding upper[{i}]"
        );
    }
    Ok(())
}

/// The gate is verdict-specific, not a global GPU kill-switch: a GPU GEMM call
/// (used by IBP / intermediate / CROWN backward GEMM) is unaffected. We assert
/// the helper only masks `as_gpu_crown_backward`, leaving `gemm_f32` callable.
#[test]
fn gate_does_not_disable_gpu_gemm() -> Result<()> {
    let _g = lock_gate();
    let gpu = PoisonedGpuCrownEngine::new(vec![0.0], vec![0.0]);
    set_sound_gpu_crown_required(true);

    // GEMM still works (here delegated to CPU, but the call is NOT gated).
    let c = gpu.gemm_f32(1, 1, 1, &[2.0], &[3.0])?;
    assert_eq!(
        c,
        vec![6.0],
        "GPU GEMM must remain available under the gate"
    );
    assert_eq!(
        gpu.gpu_calls(),
        0,
        "GEMM must not touch the GPU CROWN backward counter"
    );
    Ok(())
}
