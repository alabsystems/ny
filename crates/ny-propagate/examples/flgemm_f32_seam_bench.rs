// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! μbench for the forward-linear f32 value-GEMM seam on CPU
//! (#fl-f32-cpu-seam, CONVWALL_PANEL_VERDICT_2026-08-01 Lane A step 2).
//!
//! Compares, at the FL conv-composition census shapes:
//!   1. `f64 faer`   — the current certified f64 value-GEMM path
//!      (row-major `MatRef` views, `Accum::Replace`, global par);
//!   2. `f32 engine` — the process-global fast-f32 registry engine installed by
//!      non-CUDA startup (`install_cpu_gemm_engine_if_absent`), called through
//!      `fast_f32_gemm::with_engine` exactly as `forward_value_gemm_f32` does;
//!   3. `f32 seam`   — (2) plus the seam's real per-call overhead: narrowing
//!      both f64 operands to f32 and widening the result back.
//!
//! Run with: cargo run --release --example flgemm_f32_seam_bench -p ny-propagate

use std::time::Instant;

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef};
use ny_propagate::faer_parallelism::install_cpu_gemm_engine_if_absent;
use ny_propagate::fast_f32_gemm;

fn fill_f32(seed: &mut u64, buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let e = ((*seed >> 40) % 8) as i32 - 4;
        let mant = ((*seed >> 12) & 0x7f_ffff) as f32 / (1u32 << 23) as f32;
        let sign = if (*seed >> 3) & 1 == 1 { -1.0 } else { 1.0 };
        *v = sign * (1.0 + mant) * 2f32.powi(e);
    }
}

fn best_of<F: FnMut() -> f64>(reps: usize, mut run: F) -> (f64, f64) {
    let mut best = f64::INFINITY;
    let mut sink = 0.0f64;
    for _ in 0..reps {
        let t = Instant::now();
        sink += run();
        best = best.min(t.elapsed().as_secs_f64());
    }
    (best, sink)
}

fn main() {
    install_cpu_gemm_engine_if_absent();
    let installed = fast_f32_gemm::with_engine(|e| e.backend_provenance());
    println!(
        "registry engine: {:?} | faer global par: {:?}",
        installed,
        faer::get_global_parallelism()
    );
    println!(
        "{:>8} {:>6} {:>7} | {:>10} | {:>10} {:>10} | {:>6} {:>6}",
        "m", "k", "n", "f64 GMAC/s", "f32 GMAC/s", "seam GMAC/s", "f32 x", "seam x"
    );

    let shapes: &[(usize, usize, usize, &str)] = &[
        (25344, 1152, 99, "mission 559-GMAC-class value GEMM"),
        (65536, 1152, 128, "128ch 3x3 conv class (k=1152)"),
        (16384, 2304, 256, "256ch 3x3 conv class (k=2304)"),
        (230400, 27, 64, "first conv, k=27 bandwidth-bound"),
        (58, 1152, 25344, "1<<26-MAC skinny tile at k=1152"),
    ];
    let mut seed = 0xC0FFEEu64;
    for &(m, k, n, label) in shapes {
        let mut a32 = vec![0.0f32; m * k];
        let mut b32 = vec![0.0f32; k * n];
        fill_f32(&mut seed, &mut a32);
        fill_f32(&mut seed, &mut b32);
        let a64: Vec<f64> = a32.iter().map(|&x| f64::from(x)).collect();
        let b64: Vec<f64> = b32.iter().map(|&x| f64::from(x)).collect();
        let macs = (m as f64) * (k as f64) * (n as f64);
        let reps = if macs > 1e9 { 5 } else { 9 };

        // 1. Current certified f64 path idiom (mat_mul_f64_row_major shape).
        let (t64, s1) = {
            let am = MatRef::from_row_major_slice(&a64, m, k);
            let bm = MatRef::from_row_major_slice(&b64, k, n);
            let mut c = Mat::<f64>::zeros(m, n);
            matmul(
                &mut c,
                Accum::Replace,
                am,
                bm,
                1.0,
                faer::get_global_parallelism(),
            ); // warmup
            best_of(reps, || {
                matmul(
                    &mut c,
                    Accum::Replace,
                    am,
                    bm,
                    1.0,
                    faer::get_global_parallelism(),
                );
                c[(0, 0)]
            })
        };

        // 2. Registry engine f32 (production consumer call pattern).
        let warm = fast_f32_gemm::with_engine(|e| e.gemm_f32(m, k, n, &a32, &b32))
            .expect("registry engine installed")
            .expect("gemm_f32");
        assert_eq!(warm.len(), m * n);
        let (t32, s2) = best_of(reps, || {
            let r = fast_f32_gemm::with_engine(|e| e.gemm_f32(m, k, n, &a32, &b32))
                .unwrap()
                .unwrap();
            f64::from(r[0])
        });

        // 3. Seam-shaped: f64→f32 narrowing + engine + f64 widening
        //    (forward_value_gemm_f32's exact per-call work).
        let (tseam, s3) = best_of(reps, || {
            let na: Vec<f32> = a64.iter().map(|&x| x as f32).collect();
            let nb: Vec<f32> = b64.iter().map(|&x| x as f32).collect();
            let r = fast_f32_gemm::with_engine(|e| e.gemm_f32(m, k, n, &na, &nb))
                .unwrap()
                .unwrap();
            let w: Vec<f64> = r.into_iter().map(f64::from).collect();
            w[0]
        });

        println!(
            "{:>8} {:>6} {:>7} | {:>10.2} | {:>10.2} {:>11.2} | {:>5.2}x {:>5.2}x  # {} (sink {:.2e})",
            m,
            k,
            n,
            macs / t64 / 1e9,
            macs / t32 / 1e9,
            macs / tseam / 1e9,
            t64 / t32,
            t64 / tseam,
            label,
            s1 + s2 + s3,
        );
    }
}
