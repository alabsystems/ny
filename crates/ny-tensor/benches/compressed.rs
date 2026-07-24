// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks for f16 compressed bounds.
//!
//! Run with: cargo bench -p ny-tensor -- Compressed

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{ArrayD, IxDyn};
use ny_tensor::{BoundedTensor, CompressedBounds};

/// Benchmark compression (f32 -> f16).
fn bench_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("Compressed/Compress");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let lower = ArrayD::from_elem(IxDyn(&[size]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[size]), 1.0f32);
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("f32_to_f16", size),
            &bounds,
            |b, bounds| {
                b.iter(|| black_box(CompressedBounds::from_bounded_tensor(black_box(bounds))))
            },
        );
    }
    group.finish();
}

/// Benchmark decompression (f16 -> f32).
fn bench_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("Compressed/Decompress");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let lower = ArrayD::from_elem(IxDyn(&[size]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[size]), 1.0f32);
        let bounds = BoundedTensor::new(lower, upper).unwrap();
        let compressed = CompressedBounds::from_bounded_tensor(&bounds);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("f16_to_f32", size),
            &compressed,
            |b, compressed| b.iter(|| black_box(compressed.to_bounded_tensor().unwrap())),
        );
    }
    group.finish();
}

/// Benchmark widening for soundness.
fn bench_widen(c: &mut Criterion) {
    let mut group = c.benchmark_group("Compressed/Widen");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let lower = ArrayD::from_elem(IxDyn(&[size]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[size]), 1.0f32);
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("widen_0.001", size),
            &bounds,
            |b, bounds| {
                b.iter(|| {
                    let mut compressed = CompressedBounds::from_bounded_tensor(bounds);
                    compressed.widen_for_soundness(0.001);
                    black_box(compressed)
                })
            },
        );
    }
    group.finish();
}

/// Benchmark round-trip (compress + widen + decompress).
fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("Compressed/RoundTrip");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let lower = ArrayD::from_elem(IxDyn(&[size]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[size]), 1.0f32);
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("full_roundtrip", size),
            &bounds,
            |b, bounds| {
                b.iter(|| {
                    let mut compressed = CompressedBounds::from_bounded_tensor(bounds);
                    compressed.widen_for_soundness(0.001);
                    black_box(compressed.to_bounded_tensor().unwrap())
                })
            },
        );
    }
    group.finish();
}

/// Memory comparison: f32 vs f16.
fn bench_memory_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Compressed/Memory");

    for size in [10_000, 100_000, 1_000_000] {
        let lower = ArrayD::from_elem(IxDyn(&[size]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[size]), 1.0f32);
        let bounds = BoundedTensor::new(lower, upper).unwrap();
        let compressed = CompressedBounds::from_bounded_tensor(&bounds);

        let f32_bytes = size * 4 * 2; // f32 lower + upper
        let f16_bytes = compressed.memory_bytes();

        println!(
            "Size {}: f32={} bytes, f16={} bytes, ratio={:.1}%",
            size,
            f32_bytes,
            f16_bytes,
            (f16_bytes as f64 / f32_bytes as f64) * 100.0
        );

        // Benchmark clone operation as a proxy for memory operations
        group.throughput(Throughput::Bytes(f16_bytes as u64));
        group.bench_with_input(BenchmarkId::new("clone_f16", size), &compressed, |b, c| {
            b.iter(|| black_box(c.clone()))
        });

        group.throughput(Throughput::Bytes(f32_bytes as u64));
        group.bench_with_input(BenchmarkId::new("clone_f32", size), &bounds, |b, bounds| {
            b.iter(|| black_box(bounds.clone()))
        });
    }
    group.finish();
}

criterion_group!(
    compressed_benches,
    bench_compress,
    bench_decompress,
    bench_widen,
    bench_roundtrip,
    bench_memory_comparison,
);

criterion_main!(compressed_benches);
