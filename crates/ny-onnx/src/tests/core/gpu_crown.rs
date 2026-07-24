// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! End-to-end GPU CROWN backward timing tests on real VNN-COMP models (#3397).
//!
//! These tests load actual VNN-COMP ONNX models (not synthetic approximations)
//! and run CROWN propagation with a real GPU engine. This provides the "real
//! timing evidence" required before #3397 can be closed.
//!
//! For large models (metaroom, soundnessbench), CPU CROWN takes >900s, so we
//! validate GPU CROWN soundness via IBP comparison and concrete sampling rather
//! than GPU-vs-CPU comparison (which would require a 15+ minute test).
//!
//! Tests are skipped if benchmark data is unavailable or GPU initialization fails.
//!
//! Reference: designs/2026-03-06-gpu-crown-backward.md
//! Reference: designs/2026-03-09-issue-3397-gpu-crown-plan-cache.md

use super::*;
use ndarray::{ArrayD, IxDyn};
use ny_core::GemmEngine;
use ny_gpu::{Backend, ComputeDevice};
use ny_test_utils::workspace_root;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Resolve a path relative to the workspace root for benchmark data.
fn benchmark_path(rel: &str) -> PathBuf {
    workspace_root().join(rel)
}

/// Create a GPU ComputeDevice for the requested backend.
fn try_gpu_device(backend: Backend) -> Option<ComputeDevice> {
    ComputeDevice::new(backend).ok()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceAvailability {
    Optional,
}

fn gpu_device_for_test(
    backend: Backend,
    label: &str,
    _availability: DeviceAvailability,
) -> Option<ComputeDevice> {
    match try_gpu_device(backend) {
        Some(device) => Some(device),
        None => {
            eprintln!("{label}: SKIP: {backend:?} device not available");
            None
        }
    }
}

/// Build an epsilon-ball input from the model's input spec (strips batch dim).
fn model_input(model: &OnnxModel, eps: f32) -> (BoundedTensor, Vec<usize>) {
    let input_spec = model
        .network
        .inputs
        .first()
        .expect("model has no input spec");
    let shape: Vec<usize> = input_spec.shape[1..]
        .iter()
        .map(|&d| if d > 0 { d as usize } else { 1 })
        .collect();
    let center = ArrayD::zeros(IxDyn(&shape));
    let input = BoundedTensor::from_epsilon(center, eps).expect("BoundedTensor from_epsilon");
    (input, shape)
}

fn log_network_layers(label: &str, network: &ny_propagate::Network, input_dim: usize) {
    eprintln!(
        "{label}: {} layers, input_dim={input_dim}",
        network.layers().len()
    );
    for (i, layer) in network.layers().iter().enumerate() {
        eprintln!("  [{i}] {}", layer.layer_type());
    }
}

/// Run GPU CROWN on a large VNN-COMP model with pre-computed IBP bounds and
/// validate soundness via IBP comparison and concrete sampling. Returns the
/// GPU CROWN wall-clock time (excluding IBP collection).
///
/// Uses `propagate_crown_with_precomputed_ibp` to skip redundant internal IBP
/// collection. The deadline is set from BEFORE IBP collection to match
/// competition behavior: IBP + CROWN share the same per-instance timeout.
/// This ensures CROWN-IBP partial passes are deadline-limited just as they
/// would be in competition, preventing expensive CPU CROWN partial passes
/// on large Conv2d layers (#3397).
///
/// CPU CROWN is NOT run (it takes >900s on these models — that's the bug).
/// Instead, soundness is verified by:
/// 1. CROWN bounds must be at least as tight as IBP (standard invariant)
/// 2. Concrete center point output must fall within CROWN bounds
/// 3. Bounds must be finite and non-inverted
fn gpu_crown_timing_with_precomputed_ibp_soundness(
    model: &OnnxModel,
    eps: f32,
    gpu_device: &ComputeDevice,
    label: &str,
) -> Duration {
    let network = model
        .to_propagate_network()
        .unwrap_or_else(|e| panic!("{label}: to_propagate_network: {e}"));

    // GPU eligibility: proves the model would take the GPU fast-path.
    assert!(
        network.is_gpu_crown_eligible(),
        "{label}: model has unsupported layer types for GPU CROWN fast-path"
    );

    let (input, shape) = model_input(model, eps);
    let input_dim: usize = shape.iter().product();
    eprintln!("{label}: eps={eps}");
    log_network_layers(label, &network, input_dim);

    // Competition deadline starts NOW — before IBP collection. In competition,
    // IBP + CROWN share the same per-instance timeout (180s). IBP collection
    // consumes ~111s of this budget, leaving ~69s for CROWN. This deadline
    // pressure forces the CROWN-IBP loop to fall back to IBP for expensive
    // Conv2d layers (same behavior as competition).
    let competition_start = Instant::now();
    let competition_deadline = competition_start + Duration::from_mins(3);

    // Collect per-layer IBP bounds once. The last element is the output bounds
    // (used for soundness comparison). Pre-computed bounds are passed into CROWN
    // to skip redundant internal IBP collection (#3397).
    let ibp_layer_bounds = network
        .collect_ibp_bounds(&input)
        .unwrap_or_else(|e| panic!("{label}: collect_ibp_bounds failed: {e}"));
    let ibp_time = competition_start.elapsed();
    let ibp_bounds = ibp_layer_bounds
        .last()
        .unwrap_or_else(|| panic!("{label}: no IBP layer bounds"))
        .clone();
    eprintln!(
        "{label}: IBP={:.3}s ({} layers)",
        ibp_time.as_secs_f64(),
        ibp_layer_bounds.len()
    );

    // GPU CROWN with pre-computed IBP and the competition deadline.
    // The deadline was set before IBP, so CROWN gets only the remaining budget.
    // For metaroom: ~111s IBP → ~69s remaining → CROWN-IBP partial passes on
    // Conv2d layers hit the deadline, falling back to IBP. GPU backward then
    // runs with IBP intermediate bounds.
    let engine: &dyn GemmEngine = gpu_device;
    let gpu_start = Instant::now();
    let gpu_crown = network
        .propagate_crown_with_precomputed_ibp(
            &input,
            ibp_layer_bounds,
            Some(engine),
            Some(competition_deadline),
        )
        .unwrap_or_else(|e| panic!("{label}: GPU CROWN (precomputed IBP) failed: {e}"));
    let gpu_time = gpu_start.elapsed();
    eprintln!(
        "{label}: GPU CROWN (precomputed IBP)={:.3}s (deadline remaining={:.1}s)",
        gpu_time.as_secs_f64(),
        Duration::from_mins(3)
            .checked_sub(ibp_time)
            .expect("IBP phase exceeded the 180s competition budget")
            .as_secs_f64()
    );

    // Flatten for comparison.
    let ibp_lo = ibp_bounds.lower().as_slice().expect("ibp lower contiguous");
    let ibp_hi = ibp_bounds.upper().as_slice().expect("ibp upper contiguous");
    let gpu_lo = gpu_crown.lower().as_slice().expect("gpu lower contiguous");
    let gpu_hi = gpu_crown.upper().as_slice().expect("gpu upper contiguous");
    let output_dim = ibp_lo.len();

    assert_eq!(gpu_lo.len(), output_dim, "{label}: output dim mismatch");

    // Soundness check 1: CROWN must be at least as tight as IBP.
    for i in 0..output_dim {
        // GPU bounds must be non-inverted.
        assert!(
            gpu_lo[i] <= gpu_hi[i] + 1e-4,
            "{label}: GPU inverted at dim {i}: [{}, {}]",
            gpu_lo[i],
            gpu_hi[i],
        );
        // CROWN lower >= IBP lower (tighter or equal).
        assert!(
            gpu_lo[i] >= ibp_lo[i] - 1e-4,
            "{label}: GPU CROWN lower[{i}]={:.6} looser than IBP lower={:.6}",
            gpu_lo[i],
            ibp_lo[i],
        );
        // CROWN upper <= IBP upper (tighter or equal).
        assert!(
            gpu_hi[i] <= ibp_hi[i] + 1e-4,
            "{label}: GPU CROWN upper[{i}]={:.6} looser than IBP upper={:.6}",
            gpu_hi[i],
            ibp_hi[i],
        );
    }

    // Soundness check 2: GPU bounds must be finite.
    assert!(
        !gpu_lo.iter().any(|v| v.is_nan() || v.is_infinite()),
        "{label}: GPU CROWN lower bounds contain NaN/Inf"
    );
    assert!(
        !gpu_hi.iter().any(|v| v.is_nan() || v.is_infinite()),
        "{label}: GPU CROWN upper bounds contain NaN/Inf"
    );

    // Soundness check 3: concrete center point must fall within CROWN bounds.
    let center_input = {
        let center = ArrayD::zeros(IxDyn(&shape));
        BoundedTensor::new(center.clone(), center).expect("center BoundedTensor")
    };
    let concrete_out = network
        .propagate_ibp(&center_input)
        .unwrap_or_else(|e| panic!("{label}: concrete propagation failed: {e}"));
    let concrete_vals = concrete_out
        .lower()
        .as_slice()
        .expect("concrete lower contiguous");

    for i in 0..output_dim {
        assert!(
            concrete_vals[i] >= gpu_lo[i] - 1e-4 && concrete_vals[i] <= gpu_hi[i] + 1e-4,
            "{label}: concrete output[{i}]={:.6} outside GPU CROWN [{:.6}, {:.6}]",
            concrete_vals[i],
            gpu_lo[i],
            gpu_hi[i],
        );
    }

    // Report tightening: how much tighter is CROWN than IBP?
    let mut total_ibp_width = 0.0f64;
    let mut total_crown_width = 0.0f64;
    for i in 0..output_dim {
        total_ibp_width += (ibp_hi[i] - ibp_lo[i]) as f64;
        total_crown_width += (gpu_hi[i] - gpu_lo[i]) as f64;
    }
    let tightening = if total_ibp_width > 0.0 {
        1.0 - total_crown_width / total_ibp_width
    } else {
        0.0
    };
    eprintln!(
        "{label}: tightening={:.1}% (IBP width={:.4}, CROWN width={:.4})",
        tightening * 100.0,
        total_ibp_width,
        total_crown_width,
    );

    gpu_time
}

// ───────────────────────────────────────────────────────────────────────
// 0. Diagnostic: print layer types for VNN-COMP models
// ───────────────────────────────────────────────────────────────────────

/// Diagnostic: load metaroom 6cnn_ry and print its layer types.
/// This helps debug whether `try_extract_gpu_crown_layers` will accept the model.
#[test]
fn test_gpu_crown_metaroom_layer_diagnostic() {
    let model_path = benchmark_path(
        "benchmarks/vnncomp2023/benchmarks/metaroom/onnx/6cnn_ry_0_0_no_custom_OP.onnx",
    );
    if !model_path.exists() {
        eprintln!("SKIP: metaroom not available");
        return;
    }
    let model = load_onnx(&model_path).expect("load failed");
    let network = model.to_propagate_network().expect("convert failed");
    eprintln!("metaroom 6cnn_ry: {} layers", network.layers().len());
    for (i, layer) in network.layers().iter().enumerate() {
        eprintln!("  [{i}] {}", layer.layer_type());
    }
}

// ───────────────────────────────────────────────────────────────────────
// 1. Metaroom 6cnn_ry: the competition bottleneck (#3397)
// ───────────────────────────────────────────────────────────────────────

/// End-to-end GPU CROWN on a real metaroom 6cnn_ry ONNX model (#3397).
///
/// The 6cnn_ry architecture was blocking 17/100 metaroom instances with >900s
/// CPU CROWN backward time. GPU acceleration should bring this well within the
/// VNN-COMP 210s per-instance timeout.
///
/// Uses pre-computed IBP bounds to skip redundant internal IBP collection,
/// matching the competition pipeline where IBP is computed once and reused.
/// This isolates the GPU backward phase timing from IBP overhead.
///
/// Soundness validated via IBP comparison and concrete sampling (CPU CROWN
/// cannot serve as baseline — it takes >900s, which is the original bug).
///
/// Acceptance: GPU CROWN (with precomputed IBP) completes within 180s.
#[ntest::timeout(600000)]
#[test]
fn test_gpu_crown_real_metaroom_6cnn_ry_3397() {
    let model_path = benchmark_path(
        "benchmarks/vnncomp2023/benchmarks/metaroom/onnx/6cnn_ry_0_0_no_custom_OP.onnx",
    );
    if !model_path.exists() {
        eprintln!(
            "SKIP: metaroom benchmark data not available at {}",
            model_path.display()
        );
        return;
    }

    let gpu_device = match try_gpu_device(Backend::Wgpu) {
        Some(d) => d,
        None => {
            eprintln!("SKIP: GPU device not available");
            return;
        }
    };

    let model =
        load_onnx(&model_path).unwrap_or_else(|e| panic!("Failed to load metaroom model: {e}"));

    // GPU eligibility: proves the model would take the GPU fast-path.
    let network = model
        .to_propagate_network()
        .unwrap_or_else(|e| panic!("metaroom_6cnn_ry: to_propagate_network: {e}"));
    assert!(
        network.is_gpu_crown_eligible(),
        "metaroom_6cnn_ry: model has unsupported layer types for GPU CROWN fast-path"
    );

    // Use a representative epsilon from metaroom instances.csv.
    let gpu_time = gpu_crown_timing_with_precomputed_ibp_soundness(
        &model,
        0.00001,
        &gpu_device,
        "metaroom_6cnn_ry",
    );

    assert!(
        gpu_time.as_secs() < 180,
        "metaroom_6cnn_ry GPU CROWN (precomputed IBP) took {}s, exceeds 180s VNN-COMP timeout",
        gpu_time.as_secs(),
    );
    eprintln!(
        "metaroom_6cnn_ry: PASS — GPU CROWN (precomputed IBP) {:.3}s < 180s timeout",
        gpu_time.as_secs_f64(),
    );
}

// ───────────────────────────────────────────────────────────────────────
// 2. Soundnessbench: 384-output model (#3397)
// ───────────────────────────────────────────────────────────────────────

/// Run GPU CROWN with pre-computed per-layer IBP bounds (#3397).
///
/// Passes pre-computed per-layer IBP bounds into the CROWN pipeline to skip
/// the internal IBP forward pass. The `deadline` is an absolute Instant —
/// callers should set it relative to the competition start time (including IBP)
/// so that CROWN-IBP partial passes don't get extra time budget.
fn try_gpu_crown_with_precomputed_ibp(
    network: &ny_propagate::Network,
    input: &BoundedTensor,
    ibp_output_bounds: &BoundedTensor,
    ibp_layer_bounds: Vec<BoundedTensor>,
    gpu_device: &ComputeDevice,
    label: &str,
    deadline: Instant,
) -> Option<Duration> {
    let engine: &dyn GemmEngine = gpu_device;
    let crown_start = Instant::now();
    let gpu_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        network.propagate_crown_with_precomputed_ibp(
            input,
            ibp_layer_bounds,
            Some(engine),
            Some(deadline),
        )
    }));
    let crown_elapsed = crown_start.elapsed();
    eprintln!(
        "{label}: GPU CROWN (precomputed IBP) elapsed={:.3}s",
        crown_elapsed.as_secs_f64()
    );

    match gpu_result {
        Ok(Ok(gpu_crown)) => {
            let gpu_lo = gpu_crown.lower().as_slice().expect("gpu lower");
            let gpu_hi = gpu_crown.upper().as_slice().expect("gpu upper");
            let ibp_lo = ibp_output_bounds.lower().as_slice().expect("ibp lower");
            let ibp_hi = ibp_output_bounds.upper().as_slice().expect("ibp upper");
            for i in 0..ibp_lo.len() {
                assert!(
                    gpu_lo[i] >= ibp_lo[i] - 1e-4,
                    "{label}: GPU lower[{i}] looser than IBP"
                );
                assert!(
                    gpu_hi[i] <= ibp_hi[i] + 1e-4,
                    "{label}: GPU upper[{i}] looser than IBP"
                );
            }
            eprintln!("{label}: GPU CROWN (precomputed IBP) succeeded — bounds sound vs IBP");
            Some(crown_elapsed)
        }
        Ok(Err(e)) => {
            eprintln!("{label}: GPU CROWN (precomputed IBP) returned error: {e}");
            None
        }
        Err(_panic) => {
            eprintln!("{label}: GPU CROWN (precomputed IBP) panicked (unexpected wgpu error)");
            None
        }
    }
}

/// Assert that IBP bounds are finite and non-inverted.
fn assert_ibp_well_formed(bounds: &BoundedTensor, label: &str) {
    let lo = bounds.lower().as_slice().expect("ibp lower");
    let hi = bounds.upper().as_slice().expect("ibp upper");
    assert!(
        !lo.iter().any(|v| v.is_nan() || v.is_infinite()),
        "{label}: IBP lower NaN/Inf"
    );
    assert!(
        !hi.iter().any(|v| v.is_nan() || v.is_infinite()),
        "{label}: IBP upper NaN/Inf"
    );
    for i in 0..lo.len() {
        assert!(lo[i] <= hi[i] + 1e-4, "{label}: IBP inverted at dim {i}");
    }
}

fn assert_gpu_phase_within_budget(label: &str, elapsed: Duration, budget: Duration) {
    // Wall-clock competition budgets are asserted only under `--release`:
    // debug wall-clock measures the build profile, not the GPU backward phase
    // (same policy as the avoice wall-clock budget policy, see
    // `tests::core::avoice` module docs). Debug still reports the measurement.
    if cfg!(debug_assertions) {
        eprintln!(
            "{label}: GPU CROWN (precomputed IBP) took {:.3}s vs {:.3}s VNN-COMP backward \
             timeout — budget not asserted in debug",
            elapsed.as_secs_f64(),
            budget.as_secs_f64(),
        );
        return;
    }
    assert!(
        elapsed < budget,
        "{label}: GPU CROWN (precomputed IBP) took {:.3}s, exceeds {:.3}s VNN-COMP backward timeout",
        elapsed.as_secs_f64(),
        budget.as_secs_f64(),
    );
}

/// Compare GPU vs CPU CROWN bounds within tolerance, printing first 10 dims.
fn assert_gpu_matches_cpu(
    gpu_bounds: &BoundedTensor,
    cpu_bounds: &BoundedTensor,
    label: &str,
    tol: f32,
) {
    let cpu_lo = cpu_bounds.lower().as_slice().expect("cpu lower");
    let cpu_hi = cpu_bounds.upper().as_slice().expect("cpu upper");
    let gpu_lo = gpu_bounds.lower().as_slice().expect("gpu lower");
    let gpu_hi = gpu_bounds.upper().as_slice().expect("gpu upper");
    let output_dim = cpu_lo.len();
    assert_eq!(gpu_lo.len(), output_dim, "{label}: output dim mismatch");

    eprintln!("{label} bounds comparison (first 10 dims):");
    for i in 0..output_dim.min(10) {
        let lo_diff = (gpu_lo[i] - cpu_lo[i]).abs();
        let hi_diff = (gpu_hi[i] - cpu_hi[i]).abs();
        eprintln!(
            "  [{i}] CPU=[{:.6}, {:.6}] GPU=[{:.6}, {:.6}] diff=({:.2e}, {:.2e})",
            cpu_lo[i], cpu_hi[i], gpu_lo[i], gpu_hi[i], lo_diff, hi_diff,
        );
    }
    for i in 0..output_dim {
        let lo_diff = (gpu_lo[i] - cpu_lo[i]).abs();
        let hi_diff = (gpu_hi[i] - cpu_hi[i]).abs();
        assert!(
            lo_diff < tol,
            "{label} lower[{i}]: GPU={:.6}, CPU={:.6}, diff={:.2e}",
            gpu_lo[i],
            cpu_lo[i],
            lo_diff,
        );
        assert!(
            hi_diff < tol,
            "{label} upper[{i}]: GPU={:.6}, CPU={:.6}, diff={:.2e}",
            gpu_hi[i],
            cpu_hi[i],
            hi_diff,
        );
    }
    eprintln!("{label}: PASS — GPU CROWN matches CPU within {tol} tolerance");
}

/// Run a shipped small-model GPU CROWN parity check against CPU.
///
/// This keeps backend wiring coverage on an always-available fixture so
/// regressions do not depend on optional benchmark checkouts.
///
/// Asserts GPU eligibility: the model's layer types must all be in the
/// GPU fast-path supported set, proving the GPU path would be taken
/// when an engine is provided.
fn gpu_crown_small_model_matches_cpu(
    model_name: &str,
    backend: Backend,
    eps: f32,
    label: &str,
    tol: f32,
    availability: DeviceAvailability,
) {
    let path = require_test_model(model_name);
    let model =
        load_onnx(&path).unwrap_or_else(|e| panic!("{label}: failed to load {model_name}: {e}"));
    let gpu_device = match gpu_device_for_test(backend, label, availability) {
        Some(device) => device,
        None => return,
    };

    let network = model
        .to_propagate_network()
        .unwrap_or_else(|e| panic!("{label}: to_propagate_network failed: {e}"));

    // GPU eligibility: proves the model would take the GPU fast-path.
    assert!(
        network.is_gpu_crown_eligible(),
        "{label}: model {model_name} has unsupported layer types for GPU CROWN fast-path"
    );

    let (input, _shape) = model_input(&model, eps);

    let cpu_crown = network
        .propagate_crown(&input)
        .unwrap_or_else(|e| panic!("{label}: CPU CROWN failed: {e}"));
    let engine: &dyn GemmEngine = &gpu_device;
    let gpu_crown = network
        .propagate_crown_with_engine(&input, Some(engine))
        .unwrap_or_else(|e| panic!("{label}: GPU CROWN failed: {e}"));

    assert_gpu_matches_cpu(&gpu_crown, &cpu_crown, label, tol);
}

/// End-to-end GPU CROWN on the real soundnessbench ONNX model (#3397).
///
/// Soundnessbench has a 384-output model (Linear → Reshape → [Conv2d → ReLU]×6
/// → Flatten → Linear) that takes 915s for CPU CROWN backward.
///
/// The intermediate Conv2d activations produce A-matrices of ~724MB total,
/// exceeding wgpu's 128MB per-binding limit. The GEMM M-batching in
/// `WgpuDevice::gemm_f32()` splits these into binding-safe chunks.
///
/// Uses a 180s competition-relative deadline. This test collects IBP once up
/// front only for soundness comparison; that standalone IBP is diagnostic test
/// overhead, not part of the `#3397` backward-phase acceptance budget.
///
/// Acceptance for `#3397`: the GPU CROWN phase using pre-computed IBP must
/// succeed and finish within 180s wall-clock.
///
/// The 180s acceptance budget and the 300s watchdog are release-only
/// (debug wall-clock measures the build profile: solo debug on an M5 Max the
/// watchdog killed this test at exactly 300s, 2026-07-19). Debug runs keep
/// every soundness assertion but get an unbounded deadline.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_gpu_crown_real_soundnessbench_3397() {
    let model_path =
        benchmark_path("benchmarks/vnncomp2025/benchmarks/soundnessbench/onnx/model.onnx");
    if !model_path.exists() {
        eprintln!(
            "SKIP: soundnessbench not available at {}",
            model_path.display()
        );
        return;
    }
    let gpu_device = match try_gpu_device(Backend::Wgpu) {
        Some(d) => d,
        None => {
            eprintln!("SKIP: GPU device not available");
            return;
        }
    };

    let overall_start = Instant::now();

    let model = load_onnx(&model_path)
        .unwrap_or_else(|e| panic!("Failed to load soundnessbench model: {e}"));
    let network = model
        .to_propagate_network()
        .unwrap_or_else(|e| panic!("soundnessbench: to_propagate_network: {e}"));

    // GPU eligibility: proves the model would take the GPU fast-path.
    assert!(
        network.is_gpu_crown_eligible(),
        "soundnessbench: model has unsupported layer types for GPU CROWN fast-path"
    );

    let (input, shape) = model_input(&model, 0.001);
    let input_dim: usize = shape.iter().product();
    log_network_layers("soundnessbench", &network, input_dim);

    // Collect per-layer IBP bounds once. The final element is the output bounds
    // (for soundness validation). Pre-computed bounds are passed into CROWN to
    // skip redundant internal IBP, saving ~59s (#3397).
    let ibp_start = Instant::now();
    let ibp_layer_bounds = network
        .collect_ibp_bounds(&input)
        .unwrap_or_else(|e| panic!("soundnessbench: IBP failed: {e}"));
    let ibp_secs = ibp_start.elapsed().as_secs_f64();
    let ibp_bounds = ibp_layer_bounds
        .last()
        .expect("soundnessbench: no IBP layer bounds")
        .clone();
    eprintln!(
        "soundnessbench: IBP={ibp_secs:.3}s ({} layers)",
        ibp_layer_bounds.len()
    );

    // Use 180s deadline from overall_start (competition timeout). The deadline
    // is absolute, computed from the start of the entire pipeline (including IBP).
    // This ensures CROWN-IBP partial passes get the same deadline budget as in
    // competition — they don't get extra time just because IBP was pre-computed.
    //
    // Release-only: in debug the 3min deadline expires during the (unoptimized)
    // pipeline itself, which would fail the run before any soundness assertion
    // executes. Debug gets an effectively unbounded 24h deadline instead.
    let competition_budget = if cfg!(debug_assertions) {
        Duration::from_hours(24)
    } else {
        Duration::from_mins(3)
    };
    let competition_deadline = overall_start + competition_budget;
    let gpu_crown_elapsed = try_gpu_crown_with_precomputed_ibp(
        &network,
        &input,
        &ibp_bounds,
        ibp_layer_bounds,
        &gpu_device,
        "soundnessbench",
        competition_deadline,
    )
    .expect("soundnessbench: GPU CROWN with precomputed IBP should succeed");
    assert_gpu_phase_within_budget("soundnessbench", gpu_crown_elapsed, Duration::from_mins(3));
    assert_ibp_well_formed(&ibp_bounds, "soundnessbench");

    let total_secs = overall_start.elapsed().as_secs_f64();
    eprintln!(
        "soundnessbench: total={total_secs:.3}s (IBP={ibp_secs:.3}s, GPU_CROWN={:.3}s)",
        gpu_crown_elapsed.as_secs_f64(),
    );
}

// ───────────────────────────────────────────────────────────────────────
// 3. Small model smoke test: mnist_conv (always available)
// ───────────────────────────────────────────────────────────────────────

/// GPU CROWN smoke test on mnist_conv.onnx — validates the pipeline works
/// end-to-end on a model that is always available in tests/models/.
///
/// Unlike the VNN-COMP tests, this model is small enough to compare GPU vs CPU
/// directly (both complete in <1s).
#[ntest::timeout(60000)]
#[test]
fn test_gpu_crown_mnist_conv_smoke() {
    gpu_crown_small_model_matches_cpu(
        "mnist_conv.onnx",
        Backend::Wgpu,
        0.1,
        "mnist_conv/wgpu",
        0.01,
        DeviceAvailability::Optional,
    );
}

// ───────────────────────────────────────────────────────────────────────
// 4. ACAS-Xu: AddConstant/SubConstant extraction (#3460)
// ───────────────────────────────────────────────────────────────────────

/// GPU CROWN on ACAS-Xu model 1_1 — validates that AddConstant/SubConstant
/// layers are correctly extracted and GPU backward matches CPU backward.
///
/// ACAS-Xu models use `MatMul + Add` (not Gemm), producing:
///   SubConstant → Flatten → [Linear(no bias) → AddConstant → ReLU] × 6 → Linear → AddConstant
///
/// Before #3460 (constant-arithmetic extraction), GPU fast-path returned None
/// for these models, forcing CPU fallback. This test confirms the extraction works
/// and bounds match.
///
/// ACAS-Xu is small (5 inputs, 5 outputs, 6×50 hidden) so CPU CROWN is fast
/// enough for direct GPU-vs-CPU comparison.
#[ntest::timeout(60000)]
#[test]
fn test_gpu_crown_acasxu_addconstant_3460() {
    let model_path = benchmark_path(
        "benchmarks/vnncomp2023/benchmarks/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx",
    );
    if !model_path.exists() {
        eprintln!(
            "SKIP: ACAS-Xu benchmark data not available at {}",
            model_path.display()
        );
        return;
    }

    let gpu_device = match try_gpu_device(Backend::Wgpu) {
        Some(d) => d,
        None => {
            eprintln!("SKIP: GPU device not available");
            return;
        }
    };

    let model =
        load_onnx(&model_path).unwrap_or_else(|e| panic!("Failed to load ACAS-Xu model: {e}"));
    let network = model
        .to_propagate_network()
        .expect("to_propagate_network failed");

    // GPU eligibility: proves the model would take the GPU fast-path.
    assert!(
        network.is_gpu_crown_eligible(),
        "ACAS-Xu 1_1: model has unsupported layer types for GPU CROWN fast-path"
    );

    // Print layer types to confirm AddConstant/SubConstant are present.
    eprintln!("ACAS-Xu 1_1: {} layers", network.layers().len());
    for (i, layer) in network.layers().iter().enumerate() {
        eprintln!("  [{i}] {}", layer.layer_type());
    }

    let (input, _shape) = model_input(&model, 0.01);

    // CPU baseline.
    let cpu_crown = network.propagate_crown(&input).expect("CPU CROWN failed");

    // GPU CROWN — this is the test subject. Before #3460, this would fall back
    // to CPU because AddConstant/SubConstant were unsupported. GPU eligibility
    // (asserted above) plus real bounds here prove the AddConstant/SubConstant
    // layers run on the GPU fast-path rather than silently falling back to CPU.
    let engine: &dyn GemmEngine = &gpu_device;
    let gpu_crown = network
        .propagate_crown_with_engine(&input, Some(engine))
        .expect("GPU CROWN failed");

    // Why NOT `assert_gpu_matches_cpu(..., 0.01)`: parity to f32 noise is the
    // WRONG bar. CPU CROWN (proven-sound f64 `A·W` + certified `γ_n·S`) and the
    // GPU sound-resident backward (directed-rounding f32 on-device) each collect
    // their own CROWN-IBP intermediate bounds and therefore pick different
    // per-neuron ReLU relaxation slopes. These are two different-but-individually
    // -sound relaxations; neither dominates, so they legitimately diverge (up to
    // ~4.7e-2 here — GPU tighter on some output dims, wider on others). Demanding
    // they agree to 0.01 asserts a coincidence, not soundness. Soundness means
    // enclosing the TRUE output range — which is what we check.

    // (i) GPU CROWN must enclose the concrete forward output range over the box.
    assert_crown_encloses_acas_samples(&network, &input, &gpu_crown, 512, "ACAS-Xu 1_1 GPU");

    // (ii) GPU CROWN must be finite, non-inverted, and no looser than IBP.
    let ibp = network.propagate_ibp(&input).expect("IBP failed");
    assert_crown_finite_within_ibp(&gpu_crown, &ibp, "ACAS-Xu 1_1 GPU");

    // (iii) LOOSE parity sanity ONLY — the two sound relaxations stay in the same
    // ballpark, but cannot be held to f32 noise (0.01). 0.1 documents "same order
    // of magnitude" without re-asserting the false parity bar.
    assert_gpu_matches_cpu(&gpu_crown, &cpu_crown, "ACAS-Xu 1_1", 0.1);
}

/// splitmix64 step — deterministic, portable sampling RNG (no `rand` dep).
fn splitmix64_next(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform f32 in [0, 1) from a splitmix64 state.
fn splitmix64_unit(s: &mut u64) -> f32 {
    ((splitmix64_next(s) >> 40) as f32) / ((1u64 << 24) as f32)
}

/// Assert a CROWN bound set ENCLOSES the concrete forward output range over the
/// whole input box — the real soundness bar for a relaxation.
///
/// Samples every corner of the box (2^n) plus `n_random` deterministic-seeded
/// interior points. At each point the concrete forward pass is `propagate_ibp`
/// on the degenerate point box (`[p, p]`): IBP on a zero-width box collapses to
/// the exact forward evaluation, exactly as the CNN soundness tests do (see
/// `tests::core::cnn::assert_concrete_within_crown`). Every output must fall
/// within `[lo, hi]` (1e-4 f32 forward-noise slack). Panics on the first escape.
fn assert_crown_encloses_acas_samples(
    network: &ny_propagate::Network,
    input: &BoundedTensor,
    crown: &BoundedTensor,
    n_random: usize,
    label: &str,
) {
    let lo = crown.lower();
    let hi = crown.upper();
    let lo = lo.as_slice().expect("crown lower contiguous");
    let hi = hi.as_slice().expect("crown upper contiguous");
    let out_dim = lo.len();

    let in_lo = input.lower();
    let in_hi = input.upper();
    let in_lo = in_lo.as_slice().expect("input lower contiguous");
    let in_hi = in_hi.as_slice().expect("input upper contiguous");
    let dim = in_lo.len();
    let in_shape = input.lower().raw_dim();
    assert!(
        dim <= 24,
        "{label}: 2^{dim} corners is too many to enumerate"
    );

    // All 2^dim corners, then n_random seeded interior points.
    let n_corners = 1usize << dim;
    let mut points: Vec<Vec<f32>> = Vec::with_capacity(n_corners + n_random);
    for mask in 0..n_corners {
        points.push(
            (0..dim)
                .map(|d| {
                    if (mask >> d) & 1 == 1 {
                        in_hi[d]
                    } else {
                        in_lo[d]
                    }
                })
                .collect(),
        );
    }
    let mut seed = 0xACA5_0000_1111_2222u64;
    for _ in 0..n_random {
        points.push(
            (0..dim)
                .map(|d| in_lo[d] + splitmix64_unit(&mut seed) * (in_hi[d] - in_lo[d]))
                .collect(),
        );
    }

    const SLACK: f32 = 1e-4;
    for (pi, p) in points.iter().enumerate() {
        let pt = ArrayD::from_shape_vec(in_shape.clone(), p.clone()).expect("point shape");
        let degenerate = BoundedTensor::new(pt.clone(), pt).expect("degenerate box");
        let out = network
            .propagate_ibp(&degenerate)
            .expect("concrete forward pass");
        let vals = out.lower();
        let vals = vals.as_slice().expect("concrete output contiguous");
        for d in 0..out_dim {
            let v = vals[d];
            assert!(
                v >= lo[d] - SLACK && v <= hi[d] + SLACK,
                "{label}: UNSOUND — concrete output[{d}]={v:.6} (sample {pi}) escapes CROWN \
                 [{:.6}, {:.6}]; a sound relaxation must enclose every forward output",
                lo[d],
                hi[d],
            );
        }
    }
}

/// Assert a CROWN bound set is finite, non-inverted, and no looser than IBP.
fn assert_crown_finite_within_ibp(crown: &BoundedTensor, ibp: &BoundedTensor, label: &str) {
    let cl = crown.lower();
    let cu = crown.upper();
    let cl = cl.as_slice().expect("crown lower contiguous");
    let cu = cu.as_slice().expect("crown upper contiguous");
    let il = ibp.lower();
    let iu = ibp.upper();
    let il = il.as_slice().expect("ibp lower contiguous");
    let iu = iu.as_slice().expect("ibp upper contiguous");
    for i in 0..cl.len() {
        assert!(
            cl[i].is_finite() && cu[i].is_finite(),
            "{label}: non-finite CROWN bound at dim {i}: [{}, {}]",
            cl[i],
            cu[i],
        );
        assert!(
            cl[i] <= cu[i] + 1e-4,
            "{label}: inverted CROWN at dim {i}: [{}, {}]",
            cl[i],
            cu[i],
        );
        assert!(
            cl[i] >= il[i] - 1e-4,
            "{label}: CROWN lower[{i}]={} looser than IBP lower={}",
            cl[i],
            il[i],
        );
        assert!(
            cu[i] <= iu[i] + 1e-4,
            "{label}: CROWN upper[{i}]={} looser than IBP upper={}",
            cu[i],
            iu[i],
        );
    }
}

/// Regression: GPU CROWN and CPU CROWN each ENCLOSE the concrete ACAS-Xu output
/// range over the input box (#3460).
///
/// This is the correct soundness bar — and the reason the sibling parity test no
/// longer asserts GPU==CPU to 0.01. CPU CROWN (proven-sound f64 `A·W` + a
/// certified `γ_n·S` rounding term) and the GPU sound-resident backward
/// (directed-rounding f32 kept on-device) each run their OWN CROWN-IBP
/// intermediate-bound collection, so they pick DIFFERENT per-neuron ReLU
/// relaxation slopes. Two different-but-individually-sound relaxations do not
/// dominate each other: on this model GPU is tighter than CPU on some output
/// dims and wider on others (up to ~4.7e-2 apart). "Tighter than CPU" is NOT
/// unsoundness — soundness means enclosing the TRUE output range, which we check
/// directly by concrete sampling (32 corners + 2000 seeded interior points).
#[ntest::timeout(60000)]
#[test]
fn test_gpu_crown_acasxu_encloses_concrete_samples() {
    let model_path = benchmark_path(
        "benchmarks/vnncomp2023/benchmarks/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx",
    );
    if !model_path.exists() {
        eprintln!("SKIP: ACAS-Xu benchmark data not available");
        return;
    }
    let gpu_device = match try_gpu_device(Backend::Wgpu) {
        Some(d) => d,
        None => {
            eprintln!("SKIP: GPU device not available");
            return;
        }
    };
    let model = load_onnx(&model_path).expect("load ACAS-Xu");
    let network = model.to_propagate_network().expect("to_propagate_network");

    let (input, _shape) = model_input(&model, 0.01);
    let cpu_crown = network.propagate_crown(&input).expect("CPU CROWN");
    let engine: &dyn GemmEngine = &gpu_device;
    let gpu_crown = network
        .propagate_crown_with_engine(&input, Some(engine))
        .expect("GPU CROWN");

    // Both relaxations must enclose the true forward output range.
    assert_crown_encloses_acas_samples(&network, &input, &cpu_crown, 2000, "ACAS-Xu CPU CROWN");
    assert_crown_encloses_acas_samples(&network, &input, &gpu_crown, 2000, "ACAS-Xu GPU CROWN");
}

// ───────────────────────────────────────────────────────────────────────
// 5. dist_shift: Sigmoid model requires graph-mode (#3460)
// ───────────────────────────────────────────────────────────────────────
// Note: dist_shift mnist_generator.onnx uses Concat (multi-input op) which
// blocks both sequential CROWN and sequential IBP. GPU CROWN for Sigmoid is
// validated via synthetic networks in ny-propagate gpu_fast_path tests.
// Real-model Sigmoid GPU-vs-CPU comparison requires graph-mode CROWN (#3460).
