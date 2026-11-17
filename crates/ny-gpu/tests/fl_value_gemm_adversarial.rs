// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![recursion_limit = "256"]

//! Adversarial controls for `FlValueGemmDevice`: live binding-budget edges,
//! row chunking, chunk-partition exactness, dispatch limits and typed refusals.
//! Boundaries come from [`FlValueGemmDevice::chunk_budget_elems`].
//!
//! Regression controls cover buffers admitted by the live budget that exceed
//! the former static 128 MiB cap:
//! [`b_past_static_wall_within_live_budget_computes`] and
//! [`b_at_live_budget_edge_computes`].
//! [`rows_past_tiled_dispatch_limit_compute_exactly`] covers the tiled
//! shader's dispatch-grid cap for k > 64.
//!
//! # Gating
//!
//! Requires `wgpu,gpu-tests`; workspace tests enable these through `ny-cli`.
//! An unavailable adapter fails, as specified in `TEST_CONFORMANCE.md`.
//!
//! The consuming forward-linear f32 seam is disarmed (default off since
//! `2cd8b525f`; no shipped preset sets `forward_linear_f32_seam`), but the
//! device under test is not: it is built into the production binary and
//! its factory is registered on every wgpu host. There is no cargo feature
//! for the arm, so these tests run with the rest of the GPU suite and keep
//! the device honest before any re-arm.
//!
//! # Memory
//!
//! On a big-binding adapter the two live-edge tests move about 1.8 GB of
//! `B` (or `A`) to the device. Their host buffers start as zeroed
//! allocations with only a handful of nonzero rows written, and a local
//! mutex keeps them from overlapping even under `--test-threads` > 1.

use ny_core::{GemmEngine, NyError};
use ny_gpu::fl_value_gemm::{FL_VALUE_CHUNK_BUDGET_ELEMS, FL_VALUE_GEMM_MIN_MACS};
use ny_gpu::FlValueGemmDevice;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// wgpu's DEFAULT 128 MiB storage binding in f32 elements after the buffer
/// pool's 1.2x growth (the crate-private static `MAX_BINDING_ELEMS`). The
/// public floor [`FL_VALUE_CHUNK_BUDGET_ELEMS`] is defined as exactly half of
/// it; the assertion below catches drift in either.
const STATIC_WALL_ELEMS: usize = 2 * FL_VALUE_CHUNK_BUDGET_ELEMS;
const _: () = assert!(STATIC_WALL_ELEMS == 134_217_728 * 5 / 6 / 4);

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|poison| poison.into_inner())
}

fn require_device() -> FlValueGemmDevice {
    FlValueGemmDevice::new_wgpu().unwrap_or_else(|e| {
        panic!(
            "GPU required but not available: {e}. Run with --features gpu-tests \
             only when GPU hardware is present."
        )
    })
}

fn far_deadline() -> Instant {
    Instant::now() + Duration::from_secs(600)
}

/// Smallest row count that puts `rows * row_macs` at or above the tier's
/// measured-crossover threshold (below it the tier refuses before any budget
/// check runs).
fn rows_for_threshold(row_macs: usize) -> usize {
    let rows = FL_VALUE_GEMM_MIN_MACS.div_ceil(row_macs as u128).max(1);
    usize::try_from(rows).expect("threshold row count fits usize")
}

fn stream(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed | 1;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let e = ((s >> 40) % 8) as i32 - 4;
        let mant = ((s >> 12) & 0x7f_ffff) as f32 / (1u32 << 23) as f32;
        let sign = if (s >> 3) & 1 == 1 { -1.0 } else { 1.0 };
        sign * (1.0 + mant) * 2f32.powi(e)
    }
}

/// |gpu - f64_ref| <= gamma_{k+4}^f32 * S for one output coefficient (the
/// call site's charge).
fn assert_gamma_sound(out: &[f32], a: &[f32], b: &[f32], k: usize, n: usize, i: usize, j: usize) {
    let u = 2f64.powi(-24);
    let ku = ((k + 4) as f64) * u;
    let gamma = ku / (1.0 - ku);
    let mut refv = 0.0f64;
    let mut s = 0.0f64;
    for kk in 0..k {
        let x = f64::from(a[i * k + kk]);
        let y = f64::from(b[kk * n + j]);
        refv += x * y;
        s += x.abs() * y.abs();
    }
    let err = (f64::from(out[i * n + j]) - refv).abs();
    let bound = gamma * s * (1.0 + 1e-6);
    assert!(
        err <= bound,
        "UNSOUND at ({i},{j}): err {err} > gamma*S {bound}"
    );
}

/// Exact reference for one coefficient of an integer-valued product (every
/// partial sum is an integer far below 2^24, so the f32 GEMM must match it
/// bit for bit regardless of summation order).
fn exact_coeff(a: &[f32], b: &[f32], k: usize, n: usize, i: usize, j: usize) -> f32 {
    let mut acc = 0i64;
    for kk in 0..k {
        acc += a[i * k + kk] as i64 * b[kk * n + j] as i64;
    }
    acc as f32
}

/// ATTACK 3: `B` (k*n) one column past the handle's budget must be a typed
/// `GpuMemoryExceeded` carrying that budget — never an OOM or validation
/// panic — because `B` is shared by every row chunk and cannot be split
/// without splitting k. `B` is a zeroed allocation the refusal never reads.
#[test]
fn b_one_column_past_live_budget_is_typed_memory_refusal() {
    let _g = serial();
    let device = require_device();
    let budget = device.chunk_budget_elems();
    let k = 1024usize;
    let n = budget / k + 1;
    assert!(
        k * n > budget && k * n - budget <= k,
        "B must sit just past the budget"
    );
    let m = rows_for_threshold(k * n);
    let a = vec![1.0f32; m * k];
    let b = vec![0.0f32; k * n];
    let err = device
        .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), usize::MAX)
        .expect_err("B past the budget must refuse");
    match err {
        NyError::GpuMemoryExceeded {
            required_bytes,
            budget_bytes,
        } => {
            assert_eq!(required_bytes, k * n * 4, "refusal must name B's bytes");
            assert_eq!(
                budget_bytes,
                budget * 4,
                "refusal must name the live budget"
            );
        }
        other => panic!("unexpected refusal type: {other}"),
    }
}

/// The #hard-caps regression pin. `B` just past the static 128 MiB wall is
/// inside the live budget on any adapter that grants more than wgpu's default
/// binding size, so the tier must return the product; before the fix the
/// device's own gate refused it after the tier had admitted it. On an adapter
/// that grants only the default, the same `B` is past the tier's budget and
/// must be the typed refusal instead: both branches assert.
#[test]
fn b_past_static_wall_within_live_budget_computes() {
    let _g = serial();
    let device = require_device();
    let budget = device.chunk_budget_elems();
    let k = 1152usize;
    let n = STATIC_WALL_ELEMS / k + 1; // k*n = 27_962_496 > 27_962_026
    assert!(k * n > STATIC_WALL_ELEMS);
    // 18 rows -> 503 MMAC. Integer-valued operands: every coefficient is
    // exact in f32.
    let m = rows_for_threshold(k * n);
    let a: Vec<f32> = (0..m * k).map(|x| ((x * 7 + 3) % 3) as f32 - 1.0).collect();
    let b: Vec<f32> = (0..k * n)
        .map(|x| ((x * 13 + 5) % 5) as f32 - 2.0)
        .collect();
    let result = device.gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), usize::MAX);
    if k * n > budget {
        match result {
            Err(NyError::GpuMemoryExceeded { budget_bytes, .. }) => {
                assert_eq!(budget_bytes, budget * 4);
            }
            Err(other) => panic!("default-limit adapter: unexpected refusal type: {other}"),
            Ok(out) => panic!(
                "default-limit adapter: B past the budget must refuse, got {} values",
                out.len()
            ),
        }
        return;
    }
    let out = result.unwrap_or_else(|e| {
        panic!(
            "B of {} elements is inside the live budget ({budget}); the tier admitted it \
             so the device must compute it, got: {e}",
            k * n
        )
    });
    assert_eq!(out.len(), m * n);
    for i in [0usize, m / 2, m - 1] {
        for j in [0usize, 1, n / 2, n - 2, n - 1] {
            assert_eq!(out[i * n + j], exact_coeff(&a, &b, k, n, i, j), "({i},{j})");
        }
    }
}

/// ATTACK 3 edge: `B` at the largest column count the budget admits must be
/// computed, and computed from the WHOLE binding — the last row of `B` sits
/// at the far end of the buffer, so a truncated or short binding reads as a
/// wrong coefficient, never as agreement. `B` is zero except rows 0 and k-1.
#[test]
fn b_at_live_budget_edge_computes() {
    let _g = serial();
    let device = require_device();
    let budget = device.chunk_budget_elems();
    let k = 1024usize;
    let n = budget / k;
    assert!(
        k * n <= budget && budget - k * n < k,
        "B must sit at the budget edge"
    );
    let m = rows_for_threshold(k * n);
    let a = vec![1.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    for j in 0..n {
        b[j] = (j % 7) as f32 + 1.0; // row 0
        b[(k - 1) * n + j] = (j % 5) as f32 * 16.0; // row k-1 (end of buffer)
    }
    let out = device
        .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), usize::MAX)
        .unwrap_or_else(|e| {
            panic!(
                "B at the budget edge ({} <= {budget}) must compute, got: {e}",
                k * n
            )
        });
    assert_eq!(out.len(), m * n);
    for i in [0usize, m - 1] {
        for j in [0usize, 1, 6, n / 2, n - 1] {
            let want = (j % 7) as f32 + 1.0 + (j % 5) as f32 * 16.0;
            assert_eq!(out[i * n + j], want, "({i},{j})");
        }
    }
}

/// The live split's dispatch twin. Once `WgpuDevice::gemm_f32` sized its
/// row batches from the live limit, a chunk of more than 65535 x 16 =
/// 1,048,560 rows with k above the small-K threshold (64) reached the tiled
/// shader whole and came back as "dispatch exceeds 65535", so the tier
/// silently fell back to the CPU for shapes main computed. The device must
/// re-batch for the tiled grid: the rows on both sides of the old static seam
/// (430,185 rows at k = 65) and of the tiled limit must be exact. Integer
/// operands, so every coefficient is exact in f32.
#[test]
fn rows_past_tiled_dispatch_limit_compute_exactly() {
    let _g = serial();
    let device = require_device();
    let (k, n) = (65usize, 8usize);
    let m = rows_for_threshold(k * n).max(1_100_000);
    let a: Vec<f32> = (0..m * k).map(|x| ((x * 7 + 3) % 3) as f32 - 1.0).collect();
    let b: Vec<f32> = (0..k * n)
        .map(|x| ((x * 13 + 5) % 5) as f32 - 2.0)
        .collect();
    let out = device
        .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), usize::MAX)
        .unwrap_or_else(|e| panic!("{m}x{k}x{n} must compute, got: {e}"));
    assert_eq!(out.len(), m * n);
    for i in [0usize, 430_184, 430_185, 1_048_559, 1_048_560, m - 1] {
        for j in [0usize, n - 1] {
            assert_eq!(out[i * n + j], exact_coeff(&a, &b, k, n, i, j), "({i},{j})");
        }
    }
}

/// ATTACK 3: `A` (m*k) past the budget with `B` under it must CHUNK on the
/// budget alone (no dispatch cap) — not refuse, not panic — and every row on
/// both sides of the seam and in the ragged final chunk must be right. `A` is
/// zero except the rows checked, so zero rows next to the seam also prove no
/// chunk wrote over its neighbour.
#[test]
fn a_past_live_budget_chunks_on_budget_and_stays_exact() {
    let _g = serial();
    let device = require_device();
    let budget = device.chunk_budget_elems();
    let k = 1024usize;
    let seam = budget / k; // chunk_rows when the A side binds
    let m = seam + 3; // one full chunk + a ragged 3-row tail
    let n = rows_for_threshold(m * k).max(2); // total >= threshold
    assert!(m * k > budget, "A must exceed the budget");
    assert!(
        budget / n >= seam,
        "the A side must be the binding constraint"
    );
    let mut a = vec![0.0f32; m * k];
    let live_rows = [0usize, seam - 1, seam, m - 1];
    for &i in &live_rows {
        for kk in 0..k {
            a[i * k + kk] = ((i + kk) % 3) as f32 - 1.0;
        }
    }
    let b: Vec<f32> = (0..k * n)
        .map(|x| ((x * 11 + 1) % 5) as f32 - 2.0)
        .collect();
    let out = device
        .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), usize::MAX)
        .expect("budget-driven chunking must succeed");
    assert_eq!(out.len(), m * n);
    for i in live_rows.into_iter().chain([seam - 2, seam + 1]) {
        for j in [0usize, n / 2, n - 1] {
            assert_eq!(out[i * n + j], exact_coeff(&a, &b, k, n, i, j), "({i},{j})");
        }
    }
}

/// ATTACK 3 (unchanged from the original verify): `A` well past the static
/// floor with a random wide-magnitude stream, gamma-sound on rows at the
/// start, middle and end.
#[test]
fn a_past_static_floor_is_gamma_sound() {
    let _g = serial();
    let device = require_device();
    let (m, k, n) = (16384usize, 1152usize, 64usize); // m*k = 18.9M, ~1.2 GMAC
    assert!(m * k > FL_VALUE_CHUNK_BUDGET_ELEMS);
    let mut next = stream(0xA11A);
    let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| next()).collect();
    let out = device
        .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), usize::MAX)
        .expect("must succeed");
    assert_eq!(out.len(), m * n);
    let chunk_rows = (device.chunk_budget_elems() / k).min(m);
    for i in [0usize, chunk_rows - 1, chunk_rows.min(m - 1), m / 2, m - 1] {
        for j in [0usize, n / 2, n - 1] {
            assert_gamma_sound(&out, &a, &b, k, n, i, j);
        }
    }
}

/// ATTACK 2: the row partition must be EXACT — every output computed once.
/// The per-output dot product is one thread's sequential k-loop regardless of
/// how rows are blocked, so different chunkings of the same problem must be
/// BIT-IDENTICAL. A ragged final chunk (m % chunk_rows != 0) is forced.
#[test]
fn chunk_partition_is_bit_exact_across_chunkings() {
    let _g = serial();
    let device = require_device();
    let (m, k, n) = (2048usize, 1152usize, 512usize); // census, 1.21 GMAC
    let mut next = stream(0xC0DE);
    let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| next()).collect();
    let row_macs = k * n;

    let single = device
        .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), usize::MAX)
        .expect("single submission");
    // 1000-row chunks: 2048 = 1000 + 1000 + 48 (ragged tail).
    let ragged = device
        .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), row_macs * 1000)
        .expect("ragged chunking");
    // 8-way even chunks.
    let eight = device
        .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), m * row_macs / 8)
        .expect("8-chunk");
    assert_eq!(single.len(), m * n);
    assert!(single == ragged, "ragged chunking must be bit-identical");
    assert!(single == eight, "8-way chunking must be bit-identical");
}

/// ATTACK 2 edge: chunk_rows == 1 (one submission per output row) with a wide
/// `B`, bit-identical to the single submission; one MAC less of dispatch cap
/// than a single row needs is a typed refusal (k is never split).
#[test]
fn one_row_chunks_bit_exact_and_sub_row_cap_refuses() {
    let _g = serial();
    let device = require_device();
    let (m, k, n) = (37usize, 1152usize, 12000usize); // 37 rows -> 511 MMAC
    let mut next = stream(0x0777);
    let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| next()).collect();
    let row_macs = k * n;
    let single = device
        .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), usize::MAX)
        .expect("single submission");
    let per_row = device
        .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), row_macs)
        .expect("37 one-row submissions");
    assert!(single == per_row, "one-row chunking must be bit-identical");
    for i in [0usize, 1, m - 1] {
        assert_gamma_sound(&single, &a, &b, k, n, i, n - 1);
    }
    let err = device
        .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), row_macs - 1)
        .expect_err("row above dispatch cap must refuse");
    assert!(
        matches!(err, NyError::UnsupportedConfiguration(_)),
        "unexpected refusal type: {err}"
    );
}

/// ATTACK 1 adjunct: a GENUINE finite product at/above the 1e10 sentinel
/// magnitude must be refused (the shader clamps everything to ±1e10, so any
/// |v| >= 1e10 is indistinguishable from clamped overflow and must never be
/// published under the gamma charge).
#[test]
fn genuine_large_finite_product_refused_at_sentinel() {
    let _g = serial();
    let device = require_device();
    // 1.21 GMAC. Each dot = 1152 * (3e3)^2 = 1.037e10 >= 1e10, finite (no
    // overflow).
    let (m, k, n) = (1024usize, 1152usize, 1024usize);
    let a = vec![3.0e3f32; m * k];
    let b = vec![3.0e3f32; k * n];
    let err = device
        .gemm_f32_with_deadline(m, k, n, &a, &b, far_deadline(), usize::MAX)
        .expect_err("sentinel-magnitude finite product must refuse");
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "unexpected refusal type: {err}"
    );
}
