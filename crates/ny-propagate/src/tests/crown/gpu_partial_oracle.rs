// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU CROWN-IBP partial backward parity tests (#3599).
//!
//! These tests verify that the `try_gpu_crown_partial_backward` path in
//! `ibp.rs` correctly constructs spec matrices, dispatches to the GPU engine,
//! reshapes results, and intersects with IBP bounds.
//!
//! The `ScriptedPartialGpuCrownEngine` is a queue-based mock that asserts
//! call identity (num_specs, layer order, spec matrix values) before returning
//! pre-scripted bounds. This catches the class of bugs where the production
//! code passes wrong spec rows, wrong layer ordering, or wrong output
//! dimensions — bugs that the existing `MockGpuCrownEngine` (which returns
//! wide `[-1e6, +1e6]` for size mismatches) silently erases via IBP
//! intersection.
//!
//! Design doc: designs/2026-03-13-issue-3599-gpu-crown-ibp-partial-parity-packet.md

use super::helpers::assert_bounded_tensor_close;
use super::*;
use crate::network::ibp::NetworkIbpExt;
use ndarray::{arr1, arr2, Array2};
use ny_core::{
    GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownResult, NaiveCpuGemmEngine, NyError,
    Result,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// ScriptedPartialGpuCrownEngine
// ---------------------------------------------------------------------------

/// Expected result for a single GPU partial backward call.
enum PartialGpuResult {
    /// Return exact bounds (lower, upper).
    Bounds { lower: Vec<f32>, upper: Vec<f32> },
    /// Return an error to trigger the GPU-error fallback path.
    Error(PartialGpuFailure),
}

#[derive(Clone, Copy)]
enum PartialGpuFailure {
    UnsupportedOp,
    Device,
    Validation,
    Oom,
    Deadline,
}

/// One expected GPU call with identity assertions.
struct PartialGpuExpectation {
    num_specs: usize,
    layer_kinds: Vec<&'static str>,
    result: PartialGpuResult,
}

/// Queue-based GPU mock that asserts call identity before returning scripted
/// bounds. Panics on unexpected calls (queue underflow) or leftover
/// expectations (queue not drained).
struct ScriptedPartialGpuCrownEngine {
    calls: AtomicUsize,
    expectations: Mutex<VecDeque<PartialGpuExpectation>>,
}

impl ScriptedPartialGpuCrownEngine {
    fn new(expectations: Vec<PartialGpuExpectation>) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            expectations: Mutex::new(VecDeque::from(expectations)),
        }
    }

    fn gpu_calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// Assert all expectations were consumed. Call in test teardown.
    fn assert_all_consumed(&self) {
        let remaining = self
            .expectations
            .lock()
            .expect("expectations mutex should not be poisoned")
            .len();
        assert_eq!(
            remaining, 0,
            "ScriptedPartialGpuCrownEngine has {} unconsumed expectations",
            remaining
        );
    }
}

fn gpu_layer_kinds(layers: &[GpuCrownLayer]) -> Vec<&'static str> {
    layers
        .iter()
        .map(|layer| match layer {
            GpuCrownLayer::Linear { .. } => "Linear",
            GpuCrownLayer::Activation { .. } | GpuCrownLayer::ActivationReluDualAlpha { .. } => {
                "Activation"
            }
            GpuCrownLayer::MaxPool2d { .. } => "MaxPool2d",
            GpuCrownLayer::Conv2d { .. } => "Conv2d",
        })
        .collect()
}

impl GemmEngine for ScriptedPartialGpuCrownEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        // Delegate to CPU for layers where GPU path is skipped (e.g., layer 0
        // with partial_layers.len() < 3).
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }

    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}

impl GpuCrownBackward for ScriptedPartialGpuCrownEngine {
    fn crown_backward_gpu(
        &self,
        layers: &[GpuCrownLayer],
        _spec: &[f32],
        num_specs: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);

        let expectation = self
            .expectations
            .lock()
            .expect("expectations mutex should not be poisoned")
            .pop_front()
            .unwrap_or_else(|| {
                panic!(
                    "ScriptedPartialGpuCrownEngine: unexpected GPU call #{} (queue empty). \
                     num_specs={}, layer_kinds={:?}",
                    self.calls.load(Ordering::SeqCst),
                    num_specs,
                    gpu_layer_kinds(layers),
                )
            });

        assert_eq!(
            num_specs, expectation.num_specs,
            "ScriptedPartialGpuCrownEngine: num_specs mismatch"
        );
        assert_eq!(
            gpu_layer_kinds(layers),
            expectation.layer_kinds,
            "ScriptedPartialGpuCrownEngine: layer_kinds mismatch"
        );

        match expectation.result {
            PartialGpuResult::Bounds { lower, upper } => Ok(GpuCrownResult {
                lower_bounds: lower,
                upper_bounds: upper,
            }),
            PartialGpuResult::Error(failure) => Err(match failure {
                PartialGpuFailure::UnsupportedOp => {
                    NyError::UnsupportedOp("scripted unsupported GPU op".into())
                }
                PartialGpuFailure::Device => NyError::InternalError("scripted device loss".into()),
                PartialGpuFailure::Validation => {
                    NyError::InvalidSpec("scripted GPU validation failure".into())
                }
                PartialGpuFailure::Oom => NyError::GpuMemoryExceeded {
                    required_bytes: 2,
                    budget_bytes: 1,
                },
                PartialGpuFailure::Deadline => {
                    NyError::DeadlineExceeded("scripted GPU deadline refusal".into())
                }
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Assert that `actual[idx]` matches `expected[idx]` within tolerance for
/// every index in `indices`.
fn assert_outputs_match(
    actual: &BoundedTensor,
    expected: &BoundedTensor,
    indices: std::ops::Range<usize>,
    tol: f32,
    label: &str,
) {
    for i in indices {
        let (al, au) = (actual.lower()[[i]], actual.upper()[[i]]);
        let (el, eu) = (expected.lower()[[i]], expected.upper()[[i]]);
        assert!(
            (al - el).abs() < tol && (au - eu).abs() < tol,
            "{} output {}: actual=[{}, {}], expected=[{}, {}]",
            label,
            i,
            al,
            au,
            el,
            eu,
        );
    }
}

/// Assert that `outer` is a sound relaxation of an independently computed
/// enclosure. Equality is intentionally not required: finite authority may
/// type-refuse an opaque CPU kernel and retain the looser forward bound.
fn assert_encloses(outer: &BoundedTensor, enclosed: &BoundedTensor, tolerance: f32, label: &str) {
    assert_eq!(outer.shape(), enclosed.shape(), "{label}: shape mismatch");
    for (index, ((&outer_lower, &outer_upper), (&inner_lower, &inner_upper))) in outer
        .lower()
        .iter()
        .zip(outer.upper())
        .zip(enclosed.lower().iter().zip(enclosed.upper()))
        .enumerate()
    {
        assert!(
            outer_lower <= inner_lower + tolerance && outer_upper + tolerance >= inner_upper,
            "{label} output {index}: outer=[{outer_lower}, {outer_upper}], enclosed=[{inner_lower}, {inner_upper}]"
        );
    }
}

/// Build the shared 3-layer test prefix from the design doc.
///
/// Network: Linear(2->3) -> ReLU -> Linear(3->out_dim)
/// Input: [-1.0, -0.5] to [1.0, 0.75]
fn build_parity_network(
    w2: Array2<f32>,
    b2: ndarray::Array1<f32>,
) -> Result<(Network, BoundedTensor)> {
    let w1 = arr2(&[[0.4, -0.1], [0.2, 0.3], [-0.5, 0.7]]);
    let b1 = arr1(&[0.1, -0.2, 0.05]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1))?));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2))?));

    let input = BoundedTensor::new(
        arr1(&[-1.0, -0.5]).into_dyn(),
        arr1(&[1.0, 0.75]).into_dyn(),
    )?;

    Ok((network, input))
}

// ---------------------------------------------------------------------------
// Test A: Dense parity — CPU-vs-GPU on a 3-layer network
// ---------------------------------------------------------------------------

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_gpu_partial_dense_parity_matches_cpu_3599() -> Result<()> {
    let _g = lock_gate();
    tests::with_crown_dense_budget_mb("2048", || -> Result<()> {
        // Dense parity fixture from the design doc.
        // 2 outputs, both potentially unstable → stable_frac < 0.9 → dense mode.
        let w2 = arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]);
        let b2 = arr1(&[0.0, 0.15]);
        let (network, input) = build_parity_network(w2, b2)?;

        // Step 1: CPU oracle (engine=None).
        let cpu = network.collect_crown_ibp_bounds_with_status(&input)?;
        let cpu_final = &cpu.bounds[2]; // Layer 2 = final Linear output.

        // Step 2: Script the GPU engine to return the CPU oracle bounds.
        // Since cpu_final ⊆ IBP, intersection(IBP, cpu_final) = cpu_final.
        // This tests that the plumbing (spec construction, layer extraction,
        // reshape, intersection) preserves the GPU result correctly.
        let scripted = ScriptedPartialGpuCrownEngine::new(vec![PartialGpuExpectation {
            num_specs: 2,
            layer_kinds: vec!["Linear", "Activation", "Linear"],
            result: PartialGpuResult::Bounds {
                lower: cpu_final.lower().iter().copied().collect(),
                upper: cpu_final.upper().iter().copied().collect(),
            },
        }]);

        // Step 3: Run CROWN-IBP with the scripted GPU engine.
        let gpu =
            network.collect_crown_ibp_bounds_with_engine_and_status(&input, Some(&scripted))?;

        // Step 4: Assertions.
        assert_eq!(scripted.gpu_calls(), 1, "GPU should be called exactly once");
        scripted.assert_all_consumed();

        // Final output bounds should match CPU oracle.
        assert_bounded_tensor_close(&gpu.bounds[2], cpu_final, 1e-6, "gpu partial vs cpu");

        // Provenance for final layer should be Crown (not fallback).
        assert_eq!(
            gpu.provenance_for_layer(2),
            Some(BoundsProvenance::Crown),
            "final layer should have Crown provenance, not fallback"
        );

        // No fallback events should be recorded for the GPU path.
        assert!(
            !gpu.has_fallbacks(),
            "GPU parity run should produce no fallback events, got: {:?}",
            gpu.fallback_events
        );

        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Test B: Sparse parity — num_specs < output_dim
// ---------------------------------------------------------------------------

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_gpu_partial_sparse_parity_matches_cpu_3599() -> Result<()> {
    let _g = lock_gate();
    tests::with_crown_dense_budget_mb("2048", || -> Result<()> {
        // Sparse parity fixture from the design doc.
        // 10 outputs, exactly 1 unstable (output 0) → stable_frac = 0.9 → sparse.
        let w2 = arr2(&[
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]);
        let b2 = arr1(&[
            -0.25, 0.10, -0.10, 0.05, -0.30, 0.20, -0.10, 0.02, -0.02, 0.40,
        ]);
        let (network, input) = build_parity_network(w2, b2)?;

        // Step 1: CPU oracle.
        let cpu = network.collect_crown_ibp_bounds_with_status(&input)?;
        let cpu_final = &cpu.bounds[2];

        // The unstable output is index 0 (IBP interval crosses zero).
        // Extract just that element as the sparse GPU result.
        let sparse_lower = vec![cpu_final.lower()[[0]]];
        let sparse_upper = vec![cpu_final.upper()[[0]]];

        // Step 2: Script GPU engine with sparse expectation.
        // num_specs=1 because only 1 unstable output out of 10.
        let scripted = ScriptedPartialGpuCrownEngine::new(vec![PartialGpuExpectation {
            num_specs: 1,
            layer_kinds: vec!["Linear", "Activation", "Linear"],
            result: PartialGpuResult::Bounds {
                lower: sparse_lower,
                upper: sparse_upper,
            },
        }]);

        // Step 3: Run CROWN-IBP with the scripted GPU engine.
        let gpu =
            network.collect_crown_ibp_bounds_with_engine_and_status(&input, Some(&scripted))?;

        // Step 4: Assertions.
        assert_eq!(
            scripted.gpu_calls(),
            1,
            "GPU should be called once with sparse spec"
        );
        scripted.assert_all_consumed();

        // All 10 output bounds should match CPU oracle.
        assert_bounded_tensor_close(&gpu.bounds[2], cpu_final, 1e-6, "gpu partial vs cpu");

        // Verify the stable outputs (1..=9) specifically match IBP (untouched).
        let ibp_bounds = network.collect_ibp_bounds(&input)?;
        assert_outputs_match(&gpu.bounds[2], &ibp_bounds[2], 1..10, 1e-6, "stable IBP");

        assert_eq!(
            gpu.provenance_for_layer(2),
            Some(BoundsProvenance::Crown),
            "final layer should have Crown provenance"
        );
        assert!(
            !gpu.has_fallbacks(),
            "sparse parity run should produce no fallback events"
        );

        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Test C: Oversized spec guard — 8192 threshold
// ---------------------------------------------------------------------------

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_gpu_partial_skips_oversized_spec_guard_3599() -> Result<()> {
    let _g = lock_gate();
    // Build a network with 8193 outputs — exceeds GPU_PARTIAL_MAX_SPECS.
    // The GPU path should be skipped entirely.
    //
    // Use `with_crown_dense_budget_mb("1")` to make the CPU Dense fallback
    // fast — it will hit the memory budget guard and fall back to IBP for
    // the 8193-output layer, avoiding the expensive O(8193²) identity matrix.
    let w1 = arr2(&[[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]);
    let b1 = arr1(&[0.0, 0.0, 0.0]);

    // 8193 output neurons — all potentially unstable due to zero bias.
    let w2 = Array2::<f32>::zeros((8193, 3));
    let b2 = ndarray::Array1::<f32>::zeros(8193);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1))?));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2))?));

    let input = BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())?;

    // Script GPU engine with NO expectations — it should never be called.
    let scripted = ScriptedPartialGpuCrownEngine::new(vec![]);

    // Run CROWN-IBP with memory budget to avoid expensive CPU Dense fallback.
    tests::with_crown_dense_budget_mb("1", || -> Result<()> {
        let gpu =
            network.collect_crown_ibp_bounds_with_engine_and_status(&input, Some(&scripted))?;

        // GPU should not have been called due to the 8192 guard.
        assert_eq!(
            scripted.gpu_calls(),
            0,
            "GPU should be skipped when n_specs > 8192"
        );
        scripted.assert_all_consumed();

        // Verify result is valid (bounds exist and lower <= upper).
        assert_eq!(gpu.bounds.len(), 3, "should have bounds for all 3 layers");
        for i in 0..gpu.bounds[2].len() {
            assert!(
                gpu.bounds[2].lower().as_slice().unwrap()[i]
                    <= gpu.bounds[2].upper().as_slice().unwrap()[i],
                "output[{}]: bounds should be valid",
                i,
            );
        }

        Ok(())
    })?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Test D: Unsupported layer fallback
// ---------------------------------------------------------------------------

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_gpu_partial_falls_back_on_unsupported_layer_3599() -> Result<()> {
    let _g = lock_gate();
    // Use GELU — CPU-supported but GPU-unsupported (no extraction).
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.4, -0.1], [0.2, 0.3], [-0.5, 0.7]]),
        Some(arr1(&[0.1, -0.2, 0.05])),
    )?));
    network.add_layer(Layer::GELU(GELULayer::default()));
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]),
        Some(arr1(&[0.0, 0.15])),
    )?));

    let input = BoundedTensor::new(
        arr1(&[-1.0, -0.5]).into_dyn(),
        arr1(&[1.0, 0.75]).into_dyn(),
    )?;

    // Script GPU engine with NO expectations — unsupported layer extraction
    // returns None before reaching GPU dispatch.
    let scripted = ScriptedPartialGpuCrownEngine::new(vec![]);

    let cpu = network.collect_crown_ibp_bounds_with_status(&input)?;
    let gpu = network.collect_crown_ibp_bounds_with_engine_and_status(&input, Some(&scripted))?;

    assert_eq!(
        scripted.gpu_calls(),
        0,
        "GPU should not be called when layers are GPU-unsupported"
    );
    scripted.assert_all_consumed();

    // CPU fallback produces same result.
    assert_bounded_tensor_close(&gpu.bounds[2], &cpu.bounds[2], 1e-6, "gpu partial vs cpu");

    Ok(())
}

// ---------------------------------------------------------------------------
// Test E: GPU error fallback
// ---------------------------------------------------------------------------

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_gpu_partial_falls_back_on_gpu_error_3599() -> Result<()> {
    let _g = lock_gate();
    tests::with_crown_dense_budget_mb("2048", || -> Result<()> {
        // Reuse the dense parity network.
        let w2 = arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]);
        let b2 = arr1(&[0.0, 0.15]);
        let (network, input) = build_parity_network(w2, b2)?;

        // Script GPU engine to return an error.
        let scripted = ScriptedPartialGpuCrownEngine::new(vec![PartialGpuExpectation {
            num_specs: 2,
            layer_kinds: vec!["Linear", "Activation", "Linear"],
            result: PartialGpuResult::Error(PartialGpuFailure::UnsupportedOp),
        }]);

        let cpu = network.collect_crown_ibp_bounds_with_status(&input)?;
        let gpu =
            network.collect_crown_ibp_bounds_with_engine_and_status(&input, Some(&scripted))?;

        assert_eq!(
            scripted.gpu_calls(),
            1,
            "GPU should be attempted before falling back"
        );
        scripted.assert_all_consumed();

        // CPU fallback produces same result because try_gpu_crown_partial_backward
        // returns Ok(None) on GPU error, and the caller falls through to CPU CROWN.
        assert_bounded_tensor_close(&gpu.bounds[2], &cpu.bounds[2], 1e-6, "gpu partial vs cpu");

        // The final layer should still get Crown provenance because the CPU CROWN
        // fallback path runs after the GPU error.
        assert_eq!(
            gpu.provenance_for_layer(2),
            Some(BoundsProvenance::Crown),
            "CPU CROWN fallback should produce Crown provenance, not ForwardFallback"
        );

        Ok(())
    })
}

#[ntest::timeout(10000)]
#[test]
fn partial_gpu_runtime_refusals_all_reach_cpu_crown() -> Result<()> {
    let _g = lock_gate();
    tests::with_crown_dense_budget_mb("2048", || -> Result<()> {
        let w2 = arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]);
        let b2 = arr1(&[0.0, 0.15]);
        let (network, input) = build_parity_network(w2, b2)?;
        let cpu = network.collect_crown_ibp_bounds_with_status(&input)?;

        for (failure, label) in [
            (PartialGpuFailure::Device, "device failure"),
            (PartialGpuFailure::Validation, "validation failure"),
            (PartialGpuFailure::Oom, "GPU OOM"),
            (PartialGpuFailure::Deadline, "backend deadline refusal"),
        ] {
            let scripted = ScriptedPartialGpuCrownEngine::new(vec![PartialGpuExpectation {
                num_specs: 2,
                layer_kinds: vec!["Linear", "Activation", "Linear"],
                result: PartialGpuResult::Error(failure),
            }]);
            let actual =
                network.collect_crown_ibp_bounds_with_engine_and_status(&input, Some(&scripted))?;
            assert_eq!(scripted.gpu_calls(), 1, "{label}: GPU attempt count");
            scripted.assert_all_consumed();
            assert_bounded_tensor_close(&actual.bounds[2], &cpu.bounds[2], 1e-6, label);
            assert_eq!(
                actual.provenance_for_layer(2),
                Some(BoundsProvenance::Crown)
            );
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Test F: NaN in GPU results triggers CPU fallback (#3752)
// ---------------------------------------------------------------------------

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_gpu_partial_nan_triggers_cpu_fallback_3752() -> Result<()> {
    let _g = lock_gate();
    tests::with_crown_dense_budget_mb("2048", || -> Result<()> {
        // Fix for #3752: NaN in raw GPU bounds must discard the entire GPU result
        // and fall back to CPU CROWN. Previously, RepairStrategy::Widen converted
        // NaN → ±inf before intersection_per_element could detect it, letting
        // incorrect non-NaN siblings through into the final bounds.

        let w2 = arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]);
        let b2 = arr1(&[0.0, 0.15]);
        let (network, input) = build_parity_network(w2, b2)?;

        // CPU oracle (engine=None) — ground truth for comparison.
        let cpu = network.collect_crown_ibp_bounds_with_status(&input)?;
        let cpu_final = &cpu.bounds[2];

        // Script GPU engine to return bounds with NaN elements.
        // The NaN pre-check (#3752) should discard this entirely.
        let scripted = ScriptedPartialGpuCrownEngine::new(vec![PartialGpuExpectation {
            num_specs: 2,
            layer_kinds: vec!["Linear", "Activation", "Linear"],
            result: PartialGpuResult::Bounds {
                lower: vec![f32::NAN, -0.5],
                upper: vec![0.5, f32::NAN],
            },
        }]);

        let gpu =
            network.collect_crown_ibp_bounds_with_engine_and_status(&input, Some(&scripted))?;

        assert_eq!(scripted.gpu_calls(), 1, "GPU should be attempted");
        scripted.assert_all_consumed();

        // After the NaN pre-check, GPU result is discarded and CPU CROWN runs.
        // Final bounds should match the CPU oracle exactly.
        assert_bounded_tensor_close(&gpu.bounds[2], cpu_final, 1e-6, "gpu partial vs cpu");

        // Provenance should be Crown (from CPU CROWN, not GPU).
        assert_eq!(
            gpu.provenance_for_layer(2),
            Some(BoundsProvenance::Crown),
            "CPU fallback should produce Crown provenance"
        );

        Ok(())
    })
}

#[ntest::timeout(10000)]
#[test]
fn partial_gpu_wrong_shape_and_infinity_fall_back_to_cpu() -> Result<()> {
    let _g = lock_gate();
    tests::with_crown_dense_budget_mb("2048", || -> Result<()> {
        let w2 = arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]);
        let b2 = arr1(&[0.0, 0.15]);
        let (network, input) = build_parity_network(w2, b2)?;
        let cpu = network.collect_crown_ibp_bounds_with_status(&input)?;

        for (lower, upper, label) in [
            (vec![-1.0], vec![1.0, 1.0], "wrong lower row count"),
            (
                vec![-1.0, -1.0],
                vec![1.0, 1.0, 1.0],
                "wrong upper row count",
            ),
            (
                vec![f32::NEG_INFINITY, -1.0],
                vec![1.0, 1.0],
                "non-finite endpoint",
            ),
        ] {
            let scripted = ScriptedPartialGpuCrownEngine::new(vec![PartialGpuExpectation {
                num_specs: 2,
                layer_kinds: vec!["Linear", "Activation", "Linear"],
                result: PartialGpuResult::Bounds { lower, upper },
            }]);
            let actual =
                network.collect_crown_ibp_bounds_with_engine_and_status(&input, Some(&scripted))?;
            assert_eq!(scripted.gpu_calls(), 1, "{label}: GPU attempt count");
            scripted.assert_all_consumed();
            assert_bounded_tensor_close(&actual.bounds[2], &cpu.bounds[2], 1e-6, label);
            assert_eq!(
                actual.provenance_for_layer(2),
                Some(BoundsProvenance::Crown)
            );
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Test G: inverted GPU bounds are a whole-result refusal
// ---------------------------------------------------------------------------

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_gpu_partial_inversion_falls_back_to_cpu_3599() -> Result<()> {
    let _g = lock_gate();
    // Reuse the dense parity network.
    let w2 = arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]);
    let b2 = arr1(&[0.0, 0.15]);
    let (network, input) = build_parity_network(w2, b2)?;

    let cpu = network.collect_crown_ibp_bounds_with_status(&input)?;
    let cpu_final = &cpu.bounds[2];

    // Script GPU engine to return slightly inverted bounds for element 0.
    // Device validation must reject the whole result rather than repair it.
    let cpu_lower: Vec<f32> = cpu_final.lower().iter().copied().collect();
    let cpu_upper: Vec<f32> = cpu_final.upper().iter().copied().collect();

    // Invert element 0: make lower > upper by a tiny amount.
    let inverted_lower = vec![cpu_upper[0] + 0.001, cpu_lower[1]];
    let inverted_upper = vec![cpu_lower[0] - 0.001, cpu_upper[1]];

    let scripted = ScriptedPartialGpuCrownEngine::new(vec![PartialGpuExpectation {
        num_specs: 2,
        layer_kinds: vec!["Linear", "Activation", "Linear"],
        result: PartialGpuResult::Bounds {
            lower: inverted_lower,
            upper: inverted_upper,
        },
    }]);

    let gpu = network.collect_crown_ibp_bounds_with_engine_and_status(&input, Some(&scripted))?;

    assert_eq!(
        scripted.gpu_calls(),
        1,
        "GPU should be called despite inverted bounds"
    );
    scripted.assert_all_consumed();

    let gpu_final = &gpu.bounds[2];
    assert_bounded_tensor_close(gpu_final, cpu_final, 1e-6, "inverted GPU fallback");

    Ok(())
}

// ---------------------------------------------------------------------------
// Test: soundness-gate routing for the per-node IBP CROWN-partial backward
// (#vnncomp-gpu-crown-soundness, un-gating site #5).
//
// Before this change, under the soundness gate the per-node IBP partial path
// took `sound_gpu_crown_backward` (returns None when the gate is on) → the
// proven-sound CPU loop decided every intermediate CROWN bound. Now it routes
// through `gpu_crown_backward_route`, which under the gate dispatches the SOUND
// resident backward (`crown_backward_gpu_sound`, carrying the certified γ_n·S
// coefficient error). These tests prove the ROUTING: under the gate the SOUND
// method is dispatched and the UNSOUND `crown_backward_gpu` is never consulted.
// ---------------------------------------------------------------------------

/// The gate is a process-global; all tests that flip it — or that require the
/// default (disabled) gate so the GPU partial path is attempted at all — hold
/// the ONE shared lock in `sound_gpu_gate::test_lock`. The guard restores the
/// default on exit so other tests keep the speed-only fast path.
use crate::sound_gpu_gate::test_lock::lock_gate;

/// A partial-backward GPU engine that ADVERTISES a sound resident backward
/// (`provides_sound_gpu_crown` = true, like `WgpuDevice`). Its unsound
/// `crown_backward_gpu` is POISONED — if it ever decides a gated intermediate
/// bound the verdict would be unsound — while `crown_backward_gpu_sound` returns
/// a supplied SOUND bound. Both paths count their calls so the test can assert
/// routing.
struct PartialSoundGpuEngine {
    sound_lower: Vec<f32>,
    sound_upper: Vec<f32>,
    poisoned_lower: Vec<f32>,
    poisoned_upper: Vec<f32>,
    unsound_calls: AtomicUsize,
    sound_calls: AtomicUsize,
    honors_deadline: bool,
    deadline_writes: Mutex<Vec<Option<Instant>>>,
}

impl PartialSoundGpuEngine {
    fn deadline_writes(&self) -> Vec<Option<Instant>> {
        self.deadline_writes
            .lock()
            .expect("deadline_writes mutex should not be poisoned")
            .clone()
    }
}

impl GemmEngine for PartialSoundGpuEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        // The CPU sound fallback still needs a (sound) GEMM if it runs.
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}

impl GpuCrownBackward for PartialSoundGpuEngine {
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
    fn set_crown_backward_deadline(&self, deadline: Option<Instant>) {
        self.deadline_writes
            .lock()
            .expect("deadline_writes mutex should not be poisoned")
            .push(deadline);
    }
}

/// GATE ON: the per-node IBP partial path dispatches the verdict-relevant
/// intermediate bound to the SOUND GPU resident backward, NEVER to the unsound
/// `crown_backward_gpu`. (Un-gating site #5: closes the last CPU-fallback CROWN
/// surface under the gate.)
#[ntest::timeout(10000)]
#[test]
fn crown_ibp_partial_gate_on_routes_to_sound_gpu_backward() -> Result<()> {
    let _g = lock_gate();
    let w2 = arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]);
    let b2 = arr1(&[0.0, 0.15]);
    let (network, input) = build_parity_network(w2, b2)?;

    // CPU oracle bound for the final layer — the sound method returns it, so a
    // CPU-equal verdict proves the SOUND method (not the poisoned one) decided it.
    let cpu = network.collect_crown_ibp_bounds_with_status(&input)?;
    let cpu_final = &cpu.bounds[2];
    let sound_lower: Vec<f32> = cpu_final.lower().iter().copied().collect();
    let sound_upper: Vec<f32> = cpu_final.upper().iter().copied().collect();
    // Poison the unsound path with an over-tight (beyond-IBP) bound.
    let poisoned_lower: Vec<f32> = sound_lower
        .iter()
        .zip(&sound_upper)
        .map(|(l, u)| l + 0.9 * (0.5 * (u - l)))
        .collect();
    let poisoned_upper: Vec<f32> = sound_lower
        .iter()
        .zip(&sound_upper)
        .map(|(l, u)| u - 0.9 * (0.5 * (u - l)))
        .collect();

    let engine = PartialSoundGpuEngine {
        sound_lower: sound_lower.clone(),
        sound_upper: sound_upper.clone(),
        poisoned_lower,
        poisoned_upper,
        unsound_calls: AtomicUsize::new(0),
        sound_calls: AtomicUsize::new(0),
        honors_deadline: false,
        deadline_writes: Mutex::new(Vec::new()),
    };

    set_sound_gpu_crown_required(true);
    assert!(is_sound_gpu_crown_required());
    let gpu = network.collect_crown_ibp_bounds_with_engine_and_status(&input, Some(&engine))?;

    // Routing: SOUND method dispatched ≥1, unsound NEVER.
    assert!(
        engine.sound_calls.load(Ordering::SeqCst) >= 1,
        "gated IBP-partial path must dispatch to the SOUND GPU backward, got {} sound calls",
        engine.sound_calls.load(Ordering::SeqCst)
    );
    assert_eq!(
        engine.unsound_calls.load(Ordering::SeqCst),
        0,
        "the unsound GPU backward must NEVER decide a gated intermediate bound"
    );

    // The intermediate bound equals the (sound) bound the GPU-sound path returned
    // (intersected with the sound IBP forward bound — still sound).
    let gpu_final = &gpu.bounds[2];
    for i in 0..gpu_final.len() {
        assert!(
            gpu_final.lower()[[i]] >= sound_lower[i] - 1e-4,
            "gated lower[{i}] must come from the sound GPU bound, not the poisoned one"
        );
        assert!(
            gpu_final.upper()[[i]] <= sound_upper[i] + 1e-4,
            "gated upper[{i}] must come from the sound GPU bound, not the poisoned one"
        );
    }
    Ok(())
}

/// GATE OFF (speed-only): the IBP-partial path keeps the fast UNSOUND
/// `crown_backward_gpu` and the sound method is NOT consulted — proving the
/// un-gating is gate-scoped and the speed-only behavior is preserved.
#[ntest::timeout(10000)]
#[test]
fn crown_ibp_partial_gate_off_keeps_fast_unsound_backward() -> Result<()> {
    let _g = lock_gate();
    let w2 = arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]);
    let b2 = arr1(&[0.0, 0.15]);
    let (network, input) = build_parity_network(w2, b2)?;

    let cpu = network.collect_crown_ibp_bounds_with_status(&input)?;
    let cpu_final = &cpu.bounds[2];
    let bound_lower: Vec<f32> = cpu_final.lower().iter().copied().collect();
    let bound_upper: Vec<f32> = cpu_final.upper().iter().copied().collect();

    // Both paths return the SOUND bound here (we are not testing soundness in this
    // case, only which method is dispatched), so the run succeeds either way.
    let engine = PartialSoundGpuEngine {
        sound_lower: bound_lower.clone(),
        sound_upper: bound_upper.clone(),
        poisoned_lower: bound_lower,
        poisoned_upper: bound_upper,
        unsound_calls: AtomicUsize::new(0),
        sound_calls: AtomicUsize::new(0),
        honors_deadline: false,
        deadline_writes: Mutex::new(Vec::new()),
    };

    assert!(!is_sound_gpu_crown_required());
    let _gpu = network.collect_crown_ibp_bounds_with_engine_and_status(&input, Some(&engine))?;

    assert!(
        engine.unsound_calls.load(Ordering::SeqCst) >= 1,
        "gate-off IBP-partial path must use the fast unsound GPU backward, got {} calls",
        engine.unsound_calls.load(Ordering::SeqCst)
    );
    assert_eq!(
        engine.sound_calls.load(Ordering::SeqCst),
        0,
        "the sound GPU backward must NOT be consulted when the gate is off"
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn finite_deadline_partial_crown_skips_noncooperative_gpu_backend() -> Result<()> {
    let _g = lock_gate();
    let w2 = arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]);
    let b2 = arr1(&[0.0, 0.15]);
    let (network, input) = build_parity_network(w2, b2)?;
    let cpu = network.collect_crown_ibp_bounds_with_status(&input)?;
    let cpu_final = &cpu.bounds[2];
    // Bound but unused: the assertions below compare against `cpu`, not IBP.
    // The CALL is kept because its `?` still asserts the IBP pass succeeds on
    // this fixture before the GPU-skip path is exercised.
    let _ibp = network.collect_ibp_bounds(&input)?;
    let lower: Vec<f32> = cpu_final.lower().iter().copied().collect();
    let upper: Vec<f32> = cpu_final.upper().iter().copied().collect();
    let engine = PartialSoundGpuEngine {
        sound_lower: lower.clone(),
        sound_upper: upper.clone(),
        poisoned_lower: lower,
        poisoned_upper: upper,
        unsound_calls: AtomicUsize::new(0),
        sound_calls: AtomicUsize::new(0),
        honors_deadline: false,
        deadline_writes: Mutex::new(Vec::new()),
    };

    set_sound_gpu_crown_required(true);
    let actual = network.collect_crown_ibp_bounds_with_engine_deadline_and_status_impl(
        &input,
        Some(&engine),
        Some(Instant::now() + Duration::from_secs(30)),
    )?;

    assert_eq!(
        engine.sound_calls.load(Ordering::SeqCst),
        0,
        "a noncooperative sound GPU backend must not launch under a finite deadline"
    );
    assert_eq!(engine.unsound_calls.load(Ordering::SeqCst), 0);
    assert!(
        engine.deadline_writes().is_empty(),
        "a declined backend must not receive a deadline lease"
    );
    // Expiry-only decline semantics (docs/REGRESSION_FC_UNSAT_LOST_2026-08-14.md):
    // a LIVE finite deadline no longer refuses the dense CPU step, so the final
    // layer carries tight Crown provenance. The staging invariants above (zero
    // GPU calls, no device lease) are unchanged and remain the real contract.
    assert_eq!(
        actual.provenance_for_layer(2),
        Some(BoundsProvenance::Crown),
        "a live (unexpired) deadline must keep the tight dense CPU path"
    );
    assert_encloses(
        &actual.bounds[2],
        cpu_final,
        1e-6,
        "tight CPU result under live deadline",
    );
    assert_encloses(
        &actual.bounds[2],
        cpu_final,
        1e-6,
        "noncooperative finite fallback enclosure",
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn finite_deadline_partial_crown_skips_gpu_before_unpollable_host_setup() -> Result<()> {
    let _g = lock_gate();
    let w2 = arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]);
    let b2 = arr1(&[0.0, 0.15]);
    let (network, input) = build_parity_network(w2, b2)?;
    let cpu = network.collect_crown_ibp_bounds_with_status(&input)?;
    let cpu_final = &cpu.bounds[2];
    // Bound but unused, for the same reason as the sibling test above.
    let _ibp = network.collect_ibp_bounds(&input)?;
    let lower: Vec<f32> = cpu_final.lower().iter().copied().collect();
    let upper: Vec<f32> = cpu_final.upper().iter().copied().collect();
    let engine = PartialSoundGpuEngine {
        sound_lower: lower.clone(),
        sound_upper: upper.clone(),
        poisoned_lower: lower,
        poisoned_upper: upper,
        unsound_calls: AtomicUsize::new(0),
        sound_calls: AtomicUsize::new(0),
        honors_deadline: true,
        deadline_writes: Mutex::new(Vec::new()),
    };
    let deadline = Instant::now() + Duration::from_secs(30);

    set_sound_gpu_crown_required(true);
    let actual = network.collect_crown_ibp_bounds_with_engine_deadline_and_status_impl(
        &input,
        Some(&engine),
        Some(deadline),
    )?;

    assert_eq!(
        engine.sound_calls.load(Ordering::SeqCst),
        0,
        "finite authority must stay on the pollable CPU path before host GPU setup"
    );
    assert_eq!(engine.unsound_calls.load(Ordering::SeqCst), 0);
    assert!(
        engine.deadline_writes().is_empty(),
        "a GPU route declined before host preparation must not install a device lease"
    );
    // Expiry-only decline semantics (docs/REGRESSION_FC_UNSAT_LOST_2026-08-14.md):
    // a LIVE finite deadline no longer refuses the dense CPU step, so the final
    // layer carries tight Crown provenance. The staging invariants above (zero
    // GPU calls, no device lease) are unchanged and remain the real contract.
    assert_eq!(
        actual.provenance_for_layer(2),
        Some(BoundsProvenance::Crown),
        "a live (unexpired) deadline must keep the tight dense CPU path"
    );
    assert_encloses(
        &actual.bounds[2],
        cpu_final,
        1e-6,
        "tight CPU result under live deadline",
    );
    assert_encloses(
        &actual.bounds[2],
        cpu_final,
        1e-6,
        "cooperative finite fallback enclosure",
    );
    Ok(())
}
