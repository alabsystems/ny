// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! MUTATION VALIDATION — a probe that never fires is worthless.
//!
//! `probe_conformance.rs` shows the real Accelerate is ACCEPTED. That alone
//! proves nothing: a probe with every assertion accidentally inverted to `true`
//! would also accept it. Here the same probe code is run against pure-Rust
//! kernels carrying one deliberate defect each, and every one must be REFUSED.
//!
//! The mutants are modelled on the realistic shape of a vendor defect — a clean
//! scalar path with a defective VECTOR path — not just the easy global ones.
//! `M0` (a clean scalar kernel) is the control: it must be ACCEPTED, which is
//! what proves the probe is not simply rejecting everything.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use ny_accelerate::probe::{
    dgemm_conformance_probe_with, sgemm_conformance_probe_with, ProbeReport,
};

/// Is this call in the "blocked / vectorized kernel" regime? Used by the narrow
/// mutants, which behave perfectly on the scalar 1x1x1 path a naive probe would
/// be the only thing to test.
fn blocked(m: usize, k: usize, n: usize) -> bool {
    m >= 16 && n >= 16 && k >= 8
}

/// Clean row-major f64 kernel: sequential IEEE RN products and sums. No defect.
fn m0_clean(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) -> bool {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    true
}

fn map_kernel(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    dot: impl Fn(&[f64], &[f64], usize, usize, usize) -> f64,
) -> bool {
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = dot(a, b, i, j, k);
        }
    }
    let _ = (m, n);
    true
}

/// M1: f32 accumulation everywhere.
fn m1_f32_accum(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) -> bool {
    map_kernel(m, k, n, a, b, c, |a, b, i, j, k| {
        let mut acc = 0.0f32;
        for l in 0..k {
            acc += (a[i * k + l] as f32) * (b[l * n + j] as f32);
        }
        f64::from(acc)
    })
}

/// M2 (NARROW): f32 accumulation only for deep contractions (k >= 1024).
/// Modelled on a vendor that swaps kernels above a blocking threshold. Only
/// KA-2 (the 8x4096x8 deep-k check) can catch this.
fn m2_f32_deep_k(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) -> bool {
    if k >= 1024 {
        return m1_f32_accum(m, k, n, a, b, c);
    }
    m0_clean(m, k, n, a, b, c)
}

/// M3: flush-to-zero of subnormal results, everywhere.
fn m3_ftz(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) -> bool {
    let ok = m0_clean(m, k, n, a, b, c);
    for v in c.iter_mut() {
        if v.is_finite() && *v != 0.0 && v.abs() < f64::MIN_POSITIVE {
            *v = 0.0;
        }
    }
    ok
}

/// M4 (NARROW): FTZ only inside the blocked kernel, and only on partial sums.
/// The 1x1x1 probes see a perfect scalar path. Mutation-spec says KA-5d alone
/// catches this.
fn m4_ftz_blocked(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) -> bool {
    if !blocked(m, k, n) {
        return m0_clean(m, k, n, a, b, c);
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                let mut p = a[i * k + l] * b[l * n + j];
                if p != 0.0 && p.abs() < f64::MIN_POSITIVE {
                    p = 0.0;
                }
                acc += p;
                if acc != 0.0 && acc.abs() < f64::MIN_POSITIVE {
                    acc = 0.0;
                }
            }
            c[i * n + j] = acc;
        }
    }
    true
}

/// M5 (NARROW): denormals-are-zero on the OPERANDS, blocked kernel only.
/// Mutation-spec says KA-5e alone catches this.
fn m5_daz_blocked(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) -> bool {
    if !blocked(m, k, n) {
        return m0_clean(m, k, n, a, b, c);
    }
    let daz = |x: f64| {
        if x != 0.0 && x.abs() < f64::MIN_POSITIVE {
            0.0
        } else {
            x
        }
    };
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += daz(a[i * k + l]) * daz(b[l * n + j]);
            }
            c[i * n + j] = acc;
        }
    }
    true
}

/// M6: `beta = 0` still READS `C` (accumulates into it). A NaN-poisoned
/// destination is the only thing that catches this.
fn m6_beta0_reads_c(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) -> bool {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = acc + 0.0 * c[i * n + j];
        }
    }
    true
}

/// M7 (NARROW): `beta = 0` reads `C` only at the seam shape 32x128x32.
fn m7_beta0_reads_c_narrow(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
) -> bool {
    if (m, k, n) == (32, 128, 32) {
        return m6_beta0_reads_c(m, k, n, a, b, c);
    }
    m0_clean(m, k, n, a, b, c)
}

/// M8: saturating overflow (clamp to +/-DBL_MAX instead of infinity). This is
/// the dangerous one for NY: it converts an overflow into a large FINITE value,
/// which can silently produce a TIGHTER-than-true bound.
fn m8_saturating(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) -> bool {
    let ok = m0_clean(m, k, n, a, b, c);
    for v in c.iter_mut() {
        if *v == f64::INFINITY {
            *v = f64::MAX;
        } else if *v == f64::NEG_INFINITY {
            *v = f64::MIN;
        }
    }
    ok
}

/// M9: block mixing (a one-level Strassen-flavoured contamination). Every
/// output entry picks up an epsilon of its diagonal partner, which is exactly
/// the structure a per-entry `S`-based error bound cannot describe.
fn m9_block_mixing(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) -> bool {
    let ok = m0_clean(m, k, n, a, b, c);
    if m < 2 || n < 2 {
        return ok;
    }
    let snapshot = c.to_vec();
    for i in 0..m {
        for j in 0..n {
            let partner = snapshot[((i + 1) % m) * n + ((j + 1) % n)];
            c[i * n + j] = snapshot[i * n + j] + partner * f64::from_bits(0x3970_0000_0000_0000);
        }
    }
    ok
}

/// M10: round UPWARD on every inexact addition (emulated with an exact
/// residual test). Catches the tie check KA-4a.
fn m10_round_up(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) -> bool {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                let x = a[i * k + l] * b[l * n + j];
                let s = acc + x;
                // exact residual (Knuth two-sum)
                let bb = s - acc;
                let err = (acc - (s - bb)) + (x - bb);
                acc = if err != 0.0 && s.is_finite() {
                    // round toward +inf
                    if err > 0.0 {
                        f64::from_bits(if s >= 0.0 {
                            s.to_bits() + 1
                        } else {
                            s.to_bits() - 1
                        })
                    } else {
                        s
                    }
                } else {
                    s
                };
            }
            c[i * n + j] = acc;
        }
    }
    true
}

/// M11: truncate the product's low mantissa bits (a bf16/tf32-flavoured
/// reduced-precision multiply).
fn m11_truncated_product(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
) -> bool {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                let p = a[i * k + l] * b[l * n + j];
                acc += f64::from_bits(p.to_bits() & !0xFFFF);
            }
            c[i * n + j] = acc;
        }
    }
    true
}

type Mutant = (
    &'static str,
    fn(usize, usize, usize, &[f64], &[f64], &mut [f64]) -> bool,
);

const MUTANTS: &[Mutant] = &[
    ("M1  f32 accumulation", m1_f32_accum),
    ("M2  f32 accumulation only for k>=1024", m2_f32_deep_k),
    ("M3  flush-to-zero (global)", m3_ftz),
    ("M4  FTZ only in the blocked kernel", m4_ftz_blocked),
    ("M5  DAZ only in the blocked kernel", m5_daz_blocked),
    ("M6  beta=0 reads C (global)", m6_beta0_reads_c),
    (
        "M7  beta=0 reads C only at 32x128x32",
        m7_beta0_reads_c_narrow,
    ),
    ("M8  saturating overflow to DBL_MAX", m8_saturating),
    ("M9  Strassen-style block mixing", m9_block_mixing),
    ("M10 round upward on inexact adds", m10_round_up),
    (
        "M11 truncated (reduced-precision) product",
        m11_truncated_product,
    ),
];

#[test]
fn clean_scalar_kernel_is_accepted_by_the_probe() {
    let report = dgemm_conformance_probe_with(&m0_clean);
    assert!(
        report.accepted(),
        "M0 control REFUSED — the probe rejects a correct kernel: {:?}",
        report.failures()
    );
}

#[test]
fn every_modelled_defect_is_refused_and_the_matrix_is_recorded() {
    let mut misclassified = Vec::new();
    println!("{:<44} {:<8} caught-by", "mutant", "verdict");
    for (name, kernel) in MUTANTS {
        let report: ProbeReport =
            dgemm_conformance_probe_with(&|m, k, n, a, b, c| kernel(m, k, n, a, b, c));
        let failures = report.failures();
        println!(
            "{name:<44} {:<8} {failures:?}",
            if report.accepted() {
                "ACCEPT"
            } else {
                "REFUSE"
            }
        );
        if report.accepted() {
            misclassified.push(*name);
        }
    }
    assert!(
        misclassified.is_empty(),
        "these defects slipped through the probe: {misclassified:?}"
    );
}

/// Leave-one-out: for each check, is there a mutant ONLY it catches? This is
/// what proves the load-bearing checks (KA-5d/KA-5e/KA-2/KA-1-7/KA-6) are not
/// removable — the exact claim the probe spec makes.
#[test]
fn uniquely_load_bearing_checks_are_identified() {
    let mut unique: Vec<(&str, &str)> = Vec::new();
    for (name, kernel) in MUTANTS {
        let report = dgemm_conformance_probe_with(&|m, k, n, a, b, c| kernel(m, k, n, a, b, c));
        let failures = report.failures();
        if failures.len() == 1 {
            unique.push((name, failures[0]));
        }
    }
    println!("uniquely-caught mutants (defect -> only check that catches it):");
    for (m, c) in &unique {
        println!("  {m:<44} -> {c}");
    }
    // The narrow blocked-kernel mutants exist precisely because the prior-art
    // probe list lacked KA-5d/KA-5e; if those checks were dropped, these two
    // defects would ship.
    let caught_by = |mutant: &str| -> Vec<&'static str> {
        let (_, kernel) = MUTANTS.iter().find(|(n, _)| n.starts_with(mutant)).unwrap();
        dgemm_conformance_probe_with(&|m, k, n, a, b, c| kernel(m, k, n, a, b, c)).failures()
    };
    assert!(
        caught_by("M4").contains(&"KA-5d"),
        "KA-5d must catch blocked-kernel FTZ"
    );
    assert!(
        caught_by("M5").contains(&"KA-5e"),
        "KA-5e must catch blocked-kernel DAZ"
    );
    assert_eq!(
        caught_by("M2"),
        vec!["KA-2"],
        "the deep-k check must be the ONLY thing catching a k>=1024 precision swap"
    );
    assert_eq!(
        caught_by("M7"),
        vec!["KA-1/7"],
        "the NaN-prefilled seam-shape check must be the ONLY thing catching a narrow beta=0 read"
    );
    assert!(
        caught_by("M9").contains(&"KA-6b"),
        "KA-6 must catch block mixing"
    );
    assert!(
        caught_by("M8").contains(&"KA-8a"),
        "KA-8 must catch saturating overflow"
    );
}

// ---------------------------------------------------------------------------
// f32 probe mutation (smaller, same principle)
// ---------------------------------------------------------------------------

fn s0_clean(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) -> bool {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    true
}

fn s1_ftz(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) -> bool {
    let ok = s0_clean(m, k, n, a, b, c);
    for v in c.iter_mut() {
        if *v != 0.0 && v.abs() < f32::MIN_POSITIVE {
            *v = 0.0;
        }
    }
    ok
}

fn s2_bf16_operands(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) -> bool {
    let trunc = |x: f32| f32::from_bits(x.to_bits() & 0xFFFF_0000);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += trunc(a[i * k + l]) * trunc(b[l * n + j]);
            }
            c[i * n + j] = acc;
        }
    }
    true
}

fn s3_saturating(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) -> bool {
    let ok = s0_clean(m, k, n, a, b, c);
    for v in c.iter_mut() {
        if *v == f32::INFINITY {
            *v = f32::MAX;
        } else if *v == f32::NEG_INFINITY {
            *v = f32::MIN;
        }
    }
    ok
}

#[test]
fn f32_probe_accepts_a_clean_kernel_and_refuses_defects() {
    assert!(
        sgemm_conformance_probe_with(&s0_clean).accepted(),
        "SA control REFUSED a correct f32 kernel"
    );
    for (name, kernel) in [
        (
            "S1 f32 flush-to-zero",
            s1_ftz as fn(usize, usize, usize, &[f32], &[f32], &mut [f32]) -> bool,
        ),
        ("S2 bf16-truncated operands", s2_bf16_operands),
        ("S3 saturating overflow", s3_saturating),
    ] {
        let report = sgemm_conformance_probe_with(&|m, k, n, a, b, c| kernel(m, k, n, a, b, c));
        println!("{name:<32} {:?}", report.failures());
        assert!(!report.accepted(), "{name} slipped through the f32 probe");
    }
}
