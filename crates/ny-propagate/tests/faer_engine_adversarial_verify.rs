// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Adversarial-verify probes for the process-global faer CPU f32 engine
//! (#fl-f32-cpu-seam / #cpu-gemm-engine): degenerate and hostile shapes driven
//! through the exact public seam production consumers use
//! (`install_cpu_gemm_engine_if_absent` + `fast_f32_gemm::with_engine`).
//! Own integration binary => virgin OnceLock registry.

use ny_propagate::faer_parallelism::install_cpu_gemm_engine_if_absent;
use ny_propagate::fast_f32_gemm;

fn gemm(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> ny_core::Result<Vec<f32>> {
    fast_f32_gemm::with_engine(|e| e.gemm_f32(m, k, n, a, b)).expect("engine installed")
}

/// Cancellation-heavy wide-magnitude stream (exponents [-15, 14], random sign).
fn stream(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed | 1;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let e = ((s >> 40) % 30) as i32 - 15;
        let mant = ((s >> 12) & 0x7f_ffff) as f32 / (1u32 << 23) as f32;
        let sign = if (s >> 3) & 1 == 1 { -1.0 } else { 1.0 };
        sign * (1.0 + mant) * 2f32.powi(e)
    }
}

/// `|fl32(A·B) - exactish_f64(A·B)| <= γ_k^f32 · S` per entry, any order.
fn assert_within_envelope(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], r: &[f32]) {
    const U: f64 = 5.960_464_477_539_063e-8; // 2^-24
    let ku = (k as f64) * U;
    let gamma = ku / (1.0 - ku);
    for i in 0..m {
        for j in 0..n {
            let mut dot = 0.0f64;
            let mut s = 0.0f64;
            for kk in 0..k {
                let av = f64::from(a[i * k + kk]);
                let bv = f64::from(b[kk * n + j]);
                dot += av * bv;
                s += av.abs() * bv.abs();
            }
            let err = (f64::from(r[i * n + j]) - dot).abs();
            let bound = gamma * s * (1.0 + 1e-6) + f64::MIN_POSITIVE;
            assert!(
                err <= bound,
                "outside γ_k^f32·S envelope: err={err} bound={bound} (m={m} k={k} n={n} i={i} j={j})"
            );
        }
    }
}

#[test]
fn adversarial_shapes_stay_inside_the_f32_envelope() {
    install_cpu_gemm_engine_if_absent();
    assert!(fast_f32_gemm::is_installed());

    // --- degenerate scalars ---
    let r = gemm(1, 1, 1, &[3.5], &[-2.25]).expect("1x1x1");
    assert_eq!(r, vec![-7.875], "k=1 must be the exact single RN product");

    // m=1 / n=1 / k=1 slivers.
    let mut next = stream(0x000A_D5E1);
    for &(m, k, n) in &[
        (1usize, 1usize, 7usize),
        (7, 1, 1),
        (1, 977, 1),
        (1, 4096, 3),
        (5, 3, 1),
    ] {
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| next()).collect();
        let r = gemm(m, k, n, &a, &b).expect("sliver gemm");
        assert_eq!(r.len(), m * n);
        assert_within_envelope(m, k, n, &a, &b, &r);
    }

    // --- k = 0: empty contraction must be exactly zero, right shape ---
    let r = gemm(3, 0, 2, &[], &[]).expect("k=0");
    assert_eq!(r, vec![0.0f32; 6]);
    // m = 0 and n = 0: empty outputs.
    assert_eq!(gemm(0, 5, 4, &[], &[0.0; 20]).expect("m=0").len(), 0);
    assert_eq!(gemm(4, 5, 0, &[0.0; 20], &[]).expect("n=0").len(), 0);

    // --- non-square, prime-ish dims (blocking edge cases) ---
    for &(m, k, n) in &[(13usize, 61usize, 17usize), (3, 129, 251), (127, 33, 2)] {
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| next()).collect();
        let r = gemm(m, k, n, &a, &b).expect("non-square gemm");
        assert_within_envelope(m, k, n, &a, &b, &r);
    }

    // --- huge-k accumulation vs exact-widened f64 reference ---
    // k = 2^20 with heavy cancellation. If the engine used anything other
    // than genuine per-term f32-or-better arithmetic over exactly k products
    // (e.g. a Strassen-style recombination), the entrywise γ_k·S envelope
    // would not be guaranteed.
    let (m, k, n) = (2usize, 1usize << 20, 2usize);
    let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| next()).collect();
    let r = gemm(m, k, n, &a, &b).expect("huge-k gemm");
    assert_within_envelope(m, k, n, &a, &b, &r);

    // --- length-mismatch refusals (typed Err, no panic, no wrong shape) ---
    assert!(gemm(2, 3, 2, &[0.0; 5], &[0.0; 6]).is_err(), "short lhs");
    assert!(gemm(2, 3, 2, &[0.0; 6], &[0.0; 7]).is_err(), "long rhs");

    // --- subnormal mass: engine must not be more lossy than the FTZ budget;
    //     on a non-FTZ CPU the tiny sums survive (γ envelope trivially holds,
    //     result must be finite and non-NaN) ---
    let tiny = f32::MIN_POSITIVE / 2.0; // subnormal
    let a = vec![tiny; 64];
    let b = vec![1.0f32; 64];
    let r = gemm(1, 64, 1, &a, &b).expect("subnormal gemm");
    assert!(r[0].is_finite() && !r[0].is_nan());
    assert_within_envelope(1, 64, 1, &a, &b, &r);

    // --- overflow saturation: f32 inf may appear, must not panic ---
    let big = f32::MAX;
    let r = gemm(1, 4, 1, &[big; 4], &[big; 4]).expect("overflow gemm");
    assert!(r[0].is_infinite() && r[0] > 0.0, "must saturate to +inf");
}
