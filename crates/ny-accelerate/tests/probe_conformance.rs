// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The runtime conformance probe, run against the host's real Accelerate.
//!
//! These tests are the CI half of the "prove the vendor binary satisfies the
//! precondition" obligation: the probe that gates the engine at runtime is the
//! same code that runs here, so a host whose BLAS regresses fails the build
//! before it can fail a verdict.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use ny_accelerate::probe::{dgemm_conformance_probe, sgemm_conformance_probe, ProbeReport};

fn dump(name: &str, report: &ProbeReport) {
    println!("--- {name} ({} checks) ---", report.checks.len());
    for c in &report.checks {
        println!(
            "{:<8} {}  {}",
            c.id,
            if c.passed { "PASS" } else { "**FAIL**" },
            c.what
        );
    }
}

#[test]
fn dgemm_conformance_probe_accepts_this_host() {
    let report = dgemm_conformance_probe();
    dump("cblas_dgemm", &report);
    assert!(
        report.accepted(),
        "cblas_dgemm conformance REFUSED: {:?}",
        report.failures()
    );
    // Guard against a probe that silently stops testing things.
    assert_eq!(
        report.checks.len(),
        20,
        "probe check count changed; update the coverage record deliberately"
    );
}

#[test]
fn sgemm_conformance_probe_accepts_this_host() {
    let report = sgemm_conformance_probe();
    dump("cblas_sgemm", &report);
    assert!(
        report.accepted(),
        "cblas_sgemm conformance REFUSED: {:?}",
        report.failures()
    );
    assert_eq!(report.checks.len(), 14);
}

/// The probe must be deterministic: it is generated from a fixed LCG and makes
/// only order-free assertions, so two runs in the same process — and runs from
/// different threads — must agree exactly.
#[test]
fn probe_is_deterministic_and_thread_safe() {
    let first = dgemm_conformance_probe();
    let handles: Vec<_> = (0..8)
        .map(|_| std::thread::spawn(dgemm_conformance_probe))
        .collect();
    for h in handles {
        let r = h.join().expect("probe thread panicked");
        assert_eq!(r, first, "probe result differed across threads");
    }
    assert_eq!(dgemm_conformance_probe(), first);
}

/// Cost budget: the probe runs once per process inside the engine constructor,
/// so it must stay cheap enough to be unconditional. Spec measured 0.357 ms
/// warm / 0.483 ms cold in C; assert a generous ceiling that still catches an
/// accidental O(N^3) blow-up.
#[test]
fn probe_cost_is_bounded() {
    let cold = std::time::Instant::now();
    let _ = dgemm_conformance_probe();
    let cold = cold.elapsed();
    let mut warm = std::time::Duration::MAX;
    for _ in 0..20 {
        let t = std::time::Instant::now();
        let _ = dgemm_conformance_probe();
        warm = warm.min(t.elapsed());
    }
    println!("dgemm probe: cold {cold:?}, warm best-of-20 {warm:?}");
    assert!(
        warm < std::time::Duration::from_millis(20),
        "probe got expensive: {warm:?}"
    );
}

/// DEFAULT OFF is the moat property: with neither gate armed, nothing is
/// constructed, no probe runs, no BLAS threading policy is touched, and no
/// factory is registered — so the process is byte-identical to today.
///
/// (This test process sets no `NY_ACCELERATE_*` variables, which is exactly the
/// production default.)
#[test]
fn seam_is_default_off() {
    assert!(!ny_accelerate::f64_seam_armed());
    assert!(!ny_accelerate::f32_seam_armed());
    assert!(!ny_accelerate::kill_switch_engaged());
    let (engine, outcome) = ny_accelerate::resolve_for_install();
    assert!(engine.is_none());
    assert_eq!(outcome, ny_accelerate::InstallOutcome::NotArmed);
    assert!(ny_accelerate::AccelerateGemmEngine::new().is_none());
}
