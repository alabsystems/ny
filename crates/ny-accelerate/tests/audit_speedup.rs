// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Deterministic contracts behind the opt-in `audit_speedup` measurements.
//!
//! Timing is intentionally kept in `examples/audit_speedup.rs`: a wall-clock
//! threshold is not a portable correctness property and must not become a
//! permanently skipped test.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn armed_accelerate_executes_a_representative_exact_gemm() {
    use ny_accelerate::AccelerateGemmEngine;
    use ny_core::GemmEngine;

    // 32^3 reaches the default Accelerate dispatch floor. Integer-valued
    // operands keep the expected result exactly representable in f64.
    let dim = 32;
    let lhs = vec![1.0; dim * dim];
    let rhs = vec![1.0; dim * dim];
    let engine = AccelerateGemmEngine::new_with_gates(true, false)
        .expect("the mandatory conformance probe must admit system Accelerate");
    let result = engine
        .gemm_f64(dim, dim, dim, &lhs, &rhs)
        .expect("representative GEMM must dispatch");

    assert_eq!(result, vec![dim as f64; dim * dim]);
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[test]
fn unsupported_platform_has_a_total_inert_contract() {
    use ny_accelerate::{
        f32_seam_armed, f64_seam_armed, kill_switch_engaged, resolve_for_install, shared_engine,
        single_threaded_blas_available, telemetry, AccelerateTelemetry, InstallOutcome,
    };

    assert!(!kill_switch_engaged());
    assert!(!f64_seam_armed());
    assert!(!f32_seam_armed());
    assert!(!single_threaded_blas_available());
    assert!(shared_engine().is_none());
    assert_eq!(telemetry(), AccelerateTelemetry::default());
    let (engine, outcome) = resolve_for_install();
    assert!(engine.is_none());
    assert_eq!(outcome, InstallOutcome::NotArmed);
}
