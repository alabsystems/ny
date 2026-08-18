// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! A/B the Accelerate f64 seam against the incumbent faer kernel at
//! CROWN-shaped contractions, through the SAME `GemmEngine` entry point
//! production uses (so the G1/G2 guard scan is inside the measurement).
//!
//! `cargo run --release -p ny-accelerate --example accelerate_seam_bench`

fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    run();
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    println!("Accelerate seam is macOS/aarch64 only");
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn run() {
    use ny_accelerate::AccelerateGemmEngine;
    use ny_core::GemmEngine;
    use std::time::Instant;

    let eng = AccelerateGemmEngine::new_with_gates(true, true).expect("engine");
    println!("{}", eng.install_summary());
    println!(
        "BLASSetThreading(SINGLE_THREADED) available: {}",
        ny_accelerate::single_threaded_blas_available()
    );

    let faer_f64 =
        |m: usize, k: usize, n: usize, a: &[f64], b: &[f64], par: faer::Par| -> Vec<f64> {
            let am = faer::MatRef::from_row_major_slice(a, m, k);
            let bm = faer::MatRef::from_row_major_slice(b, k, n);
            let mut c = faer::Mat::<f64>::zeros(m, n);
            faer::linalg::matmul::matmul(&mut c, faer::Accum::Replace, am, bm, 1.0, par);
            let mut out = vec![0.0f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    out[i * n + j] = c[(i, j)];
                }
            }
            out
        };

    // Two faer baselines, because the honest comparison depends on where the
    // GEMM runs: `faer_parallelism::current_par()` forces `Par::Seq` inside a
    // rayon worker (the per-domain CROWN collection, i.e. the hot path) and
    // uses the global rayon pool outside one (the root fold).
    println!(
        "\n{:>18}  {:>10}  {:>10}  {:>10}  {:>9}  {:>9}  {:>10}",
        "shape (m,k,n)", "faerSeq ms", "faerPar ms", "accel ms", "vs Seq", "vs Par", "accel GF/s"
    );
    for &(m, k, n) in &[
        (512usize, 64usize, 512usize),
        (1024, 128, 1024),
        (2048, 256, 2048),
        (128, 128, 128),
        (256, 512, 256),
        (64, 2048, 64),
    ] {
        let mut s: u64 = 0x1234;
        let mut next = || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let a: Vec<f64> = (0..m * k).map(|_| next()).collect();
        let b: Vec<f64> = (0..k * n).map(|_| next()).collect();

        let reps = if m * k * n > 100_000_000 { 5 } else { 30 };
        // warm
        let _ = faer_f64(m, k, n, &a, &b, faer::Par::Seq);
        let _ = faer_f64(m, k, n, &a, &b, faer::get_global_parallelism());
        let _ = eng.gemm_f64(m, k, n, &a, &b).expect("accel");

        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(faer_f64(m, k, n, &a, &b, faer::Par::Seq));
        }
        let seq_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(faer_f64(m, k, n, &a, &b, faer::get_global_parallelism()));
        }
        let par_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(eng.gemm_f64(m, k, n, &a, &b).expect("accel"));
        }
        let acc_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        println!(
            "{:>18}  {seq_ms:>10.3}  {par_ms:>10.3}  {acc_ms:>10.3}  {:>8.2}x  {:>8.2}x  {:>10.1}",
            format!("{m},{k},{n}"),
            seq_ms / acc_ms,
            par_ms / acc_ms,
            flops / (acc_ms * 1e-3) / 1e9,
        );
    }

    // Guard cost: how much of the call is the G2 domain scan?
    let (m, k, n) = (512usize, 64usize, 512usize);
    let a = vec![1.5f64; m * k];
    let b = vec![0.5f64; k * n];
    let reps = 200;
    let t = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(eng.gemm_f64(m, k, n, &a, &b).expect("accel"));
    }
    println!(
        "\nfull engine call incl. G1+G2 guards at 512x64x512: {:.4} ms",
        t.elapsed().as_secs_f64() * 1e3 / f64::from(reps)
    );
    println!("{:?}", ny_accelerate::telemetry());
}
