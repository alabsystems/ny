// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! End-to-end benchmark for the explicitly qualified WGPU CROWN policy.
//!
//! The benchmark consumes the same typed request as production and registers
//! measurements only when all five live rungs qualify that exact device. Its
//! label is deliberately "end_to_end": unsupported sub-operations retain the
//! production CPU fallback, while the qualified CROWN-only seam remains the
//! sole WGPU verdict route. This is not a kernel-only timing.

use criterion::{criterion_group, criterion_main, Criterion};
use ny_gpu::benchmark_support::crown_backward_cases::build_bench_cases;
use ny_gpu::{ComputeDevice, WgpuVerdictRequest};

fn bench_wgpu_crown(criterion: &mut Criterion) {
    let device = match ComputeDevice::new_for_proof(WgpuVerdictRequest::new()) {
        Ok(device) => device,
        Err(error) => {
            eprintln!(
                "WGPU CROWN benchmark unavailable: typed qualification refused; \
                 fallback=cpu report={:?}",
                error.report()
            );
            return;
        }
    };
    let cases = build_bench_cases().expect("representative CROWN benchmark cases must build");
    let mut group = criterion.benchmark_group("qualified_wgpu_crown_end_to_end");
    for case in &cases {
        group.bench_function(case.name(), |bencher| {
            bencher.iter(|| {
                case.run_graph_crown(Some(&device))
                    .expect("qualified WGPU CROWN benchmark case must complete");
            });
        });
    }
    group.finish();
}

criterion_group!(wgpu_benches, bench_wgpu_crown);
criterion_main!(wgpu_benches);
