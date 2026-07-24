// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, measurement::WallTime, BatchSize, BenchmarkGroup, BenchmarkId};
use ny_core::GemmEngine;
use ny_gpu::benchmark_support::crown_backward_cases::{clear_gpu_crown_working_set, BenchCase};
use ny_gpu::ComputeDevice;

fn supports_phase_breakdown(case: &BenchCase) -> bool {
    matches!(
        case.name(),
        "soundnessbench_exact_like" | "metaroom_6cnn_ry_like"
    )
}

pub(crate) fn register_phase_breakdown_benchmarks(
    group: &mut BenchmarkGroup<'_, WallTime>,
    case: &BenchCase,
    gpu: &ComputeDevice,
    engine: &dyn GemmEngine,
    backend_label: &str,
) {
    if !supports_phase_breakdown(case) {
        return;
    }

    // #3397 phase split for the Conv-heavy workloads:
    // - `ibp_forward`: standalone IBP collection (CPU, backend-independent)
    // - `{backend}_crown_ibp_from_ibp`: CROWN-IBP collection only, with IBP done in setup
    // - `{backend}_production_from_ibp`: full production CROWN from precomputed IBP.
    group.bench_with_input(
        BenchmarkId::new("ibp_forward", case.name()),
        case,
        |bench, case| {
            bench.iter(|| {
                black_box(case)
                    .collect_ibp()
                    .expect("benchmark IBP collection should succeed")
            })
        },
    );

    let crown_ibp_id = format!("{backend_label}_crown_ibp_from_ibp");
    group.bench_with_input(
        BenchmarkId::new(&crown_ibp_id, case.name()),
        case,
        |bench, case| {
            bench.iter_batched(
                || {
                    clear_gpu_crown_working_set(gpu)
                        .expect("benchmark CROWN working-set cleanup should succeed")
                },
                |()| {
                    black_box(case).run_crown_ibp_from_fresh_ibp(engine).expect(
                        "benchmark CROWN-IBP collection from precomputed IBP should succeed",
                    )
                },
                BatchSize::SmallInput,
            )
        },
    );

    let production_ibp_id = format!("{backend_label}_production_from_ibp");
    group.bench_with_input(
        BenchmarkId::new(&production_ibp_id, case.name()),
        case,
        |bench, case| {
            bench.iter_batched(
                || {
                    clear_gpu_crown_working_set(gpu)
                        .expect("benchmark CROWN working-set cleanup should succeed")
                },
                |()| {
                    black_box(case)
                        .run_production_from_fresh_ibp(engine)
                        .expect("benchmark GPU CROWN from precomputed IBP should succeed")
                },
                BatchSize::SmallInput,
            )
        },
    );
}
