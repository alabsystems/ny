// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmark: production CPU vs wgpu GPU CROWN propagation (#3397, #3603).
//!
//! This benchmark intentionally measures the real `Network::propagate_crown`
//! path, not a toy scalar CPU reference. That keeps the reported speedup tied
//! to the production verifier path instead of the previous triple-loop CPU
//! surrogate that overstated GPU wins.

#[path = "crown_backward/phase_breakdown.rs"]
mod crown_backward_phase_breakdown;

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use ny_core::GemmEngine;
use ny_gpu::benchmark_support::crown_backward_cases::{
    build_bench_cases, clear_gpu_crown_working_set, cpu_crown_dense_budget_bytes, BenchCase,
};
use ny_gpu::{Backend, ComputeDevice};
use std::time::Duration;
use tracing::{error, warn};

/// Benchmark a single case with CPU and wgpu GPU backend.
fn benchmark_case(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    case: &BenchCase,
    gpu_device: Option<&ComputeDevice>,
    gpu_label: &str,
    cpu_done: &mut bool,
) {
    // Release GPU working set between model cases to prevent cross-model
    // memory accumulation (#3515 Phase 3).
    if let Some(gpu) = gpu_device {
        clear_gpu_crown_working_set(gpu)
            .expect("benchmark CROWN working-set cleanup should succeed");
    }

    group.throughput(Throughput::Elements(case.parameter_count() as u64));

    // Skip CPU CROWN for cases whose estimated Dense A-matrix peak exceeds the
    // same sequential CROWN dense-materialization budget that production uses.
    // Once the runtime guard falls back to IBP, benchmarking that path is not a
    // useful CPU-vs-GPU CROWN comparison.
    let cpu_budget = cpu_crown_dense_budget_bytes();
    let run_cpu = case.estimated_cpu_peak_bytes() <= cpu_budget && !*cpu_done;
    if !run_cpu && !*cpu_done {
        warn!(
            case = case.name(),
            estimated_gb = case.estimated_cpu_peak_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
            budget_gb = cpu_budget as f64 / (1024.0 * 1024.0 * 1024.0),
            "skipping CPU CROWN — estimated peak exceeds budget (#3515)",
        );
    }

    if run_cpu {
        case.run_cpu_production()
            .expect("production CPU CROWN should succeed");

        if let Some(gpu) = gpu_device {
            let engine: &dyn GemmEngine = gpu;
            case.assert_gpu_matches_cpu(gpu, engine, 1e-2)
                .expect("production GPU CROWN should match CPU bounds");
            clear_gpu_crown_working_set(gpu)
                .expect("benchmark CROWN working-set cleanup should succeed");
        }

        group.bench_with_input(
            BenchmarkId::new("cpu_production", case.name()),
            case,
            |b, case| {
                b.iter(|| {
                    black_box(case)
                        .run_cpu_production()
                        .expect("production CPU CROWN should succeed")
                })
            },
        );
        *cpu_done = true;
    }

    if let Some(gpu) = gpu_device {
        let engine: &dyn GemmEngine = gpu;
        let bench_id = format!("{gpu_label}_production");
        group.bench_with_input(BenchmarkId::new(&bench_id, case.name()), case, |b, case| {
            b.iter_batched(
                || {
                    clear_gpu_crown_working_set(gpu)
                        .expect("benchmark CROWN working-set cleanup should succeed")
                },
                |()| {
                    black_box(case)
                        .run_gpu_production(engine)
                        .expect("production GPU CROWN should succeed")
                },
                BatchSize::SmallInput,
            )
        });

        crown_backward_phase_breakdown::register_phase_breakdown_benchmarks(
            group, case, gpu, engine, gpu_label,
        );
    }
}

/// Benchmark wgpu backend on all cases.
fn bench_wgpu_crown(c: &mut Criterion) {
    let mut group = c.benchmark_group("GpuKernels/CrownBackwardWgpu");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let wgpu_device = ComputeDevice::new(Backend::Wgpu).ok();

    let Ok(cases) = build_bench_cases() else {
        error!("Skipping wgpu crown benchmark: unable to build cases");
        return;
    };

    for case in &cases {
        let mut cpu_done = false;
        benchmark_case(
            &mut group,
            case,
            wgpu_device.as_ref(),
            "wgpu",
            &mut cpu_done,
        );
    }

    group.finish();
}

criterion_group!(wgpu_benches, bench_wgpu_crown);
criterion_main!(wgpu_benches);
