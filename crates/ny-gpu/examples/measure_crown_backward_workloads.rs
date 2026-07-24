// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//! One-shot measurement runner for #3397 representative workloads.
//!
//! Includes both sequential `Network` phases (GPU fast-path) and optional
//! `GraphNetwork` phases (per-node CROWN-IBP loop) for the same workloads,
//! quantifying the benchmark-reality gap identified in #3716.
//!
//! Run with:
//! `cargo run -p ny-gpu --release --example measure_crown_backward_workloads`
//! Optional filter:
//! `cargo run -p ny-gpu --release --example measure_crown_backward_workloads -- --case soundnessbench_exact_like`
//! Write a clean CSV artifact directly from the example process:
//! `cargo run -p ny-gpu --release --example measure_crown_backward_workloads -- --output reports/benchmarks/gpu_crown_backward_timing_current.csv`
//! Add `--production-only` to skip the IBP/CROWN-IBP pipeline and measure just
//! the warm production/profile path. This is useful when an intermediate phase
//! is intentionally known-broken but the device profiling path still needs
//! fresh artifacts on current HEAD.
//! Add `--graph` to include GraphNetwork measurement phases (#3716).
//! These convert each sequential workload to `GraphNetwork` and measure the
//! graph CROWN-IBP collection path that real VNN-COMP ONNX models exercise.
//! Add `--graph-engine-only` to keep the graph IBP and engine-accelerated
//! collection rows while skipping the known-slow CPU graph collection lane when
//! refreshing GPU evidence.
//! Add `--graph-full --case <name>` for the extremely slow full graph CROWN
//! backward (43+ min for metaroom) — this is NOT the VNN-COMP code path but
//! useful for analysis.

use ny_core::{GemmEngine, NyError, Result};
use ny_gpu::benchmark_support::crown_backward_cases::{
    build_bench_cases, clear_gpu_crown_working_set, cpu_crown_dense_budget_bytes, BenchCase,
};
use ny_gpu::benchmark_support::crown_backward_measurements::{
    create_output_file, measure_or_skip_cpu_phase, write_csv_header, write_csv_row,
    write_failed_phase, write_measured_phase, CpuPhaseOutcome, MeasurementArgs, MeasurementRow,
};
use ny_gpu::benchmark_support::crown_backward_profiles::{
    write_gpu_profile_rows, write_host_profile_rows,
};
use ny_gpu::{Backend, ComputeDevice};
use std::io::{self, Write};
use std::panic::{self, AssertUnwindSafe};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GraphMeasurementOptions {
    include_cpu_collection: bool,
    include_full_crown: bool,
}

fn measure_seconds<T>(f: impl FnOnce() -> Result<T>) -> Result<f64> {
    let start = Instant::now();
    let _ = f()?;
    Ok(start.elapsed().as_secs_f64())
}

fn measure_cpu_phase(
    out: &mut impl Write,
    case: &BenchCase,
    gpu_device: &ComputeDevice,
    engine: &dyn GemmEngine,
    cpu_budget: usize,
) -> Result<()> {
    let outcome = measure_or_skip_cpu_phase(out, case, cpu_budget, || {
        measure_seconds(|| case.run_cpu_production())
    })?;
    if matches!(outcome, CpuPhaseOutcome::Measured) {
        clear_gpu_crown_working_set(gpu_device)?;
        case.assert_gpu_matches_cpu(gpu_device, engine, 1e-2)?;
    }
    Ok(())
}

fn measure_ibp_phase(out: &mut impl Write, case: &BenchCase, cpu_budget: usize) -> Result<()> {
    let seconds = measure_seconds(|| case.collect_ibp())?;
    write_measured_phase(out, case, "ibp_forward", seconds, cpu_budget)
}

fn measure_crown_ibp_phase(
    out: &mut impl Write,
    case: &BenchCase,
    gpu_device: &ComputeDevice,
    engine: &dyn GemmEngine,
    cpu_budget: usize,
) -> Result<()> {
    clear_gpu_crown_working_set(gpu_device)?;
    let seconds = measure_seconds(|| case.run_crown_ibp_from_fresh_ibp(engine))?;
    write_measured_phase(out, case, "wgpu_crown_ibp_from_ibp", seconds, cpu_budget)
}

fn measure_production_from_ibp_phase(
    out: &mut impl Write,
    case: &BenchCase,
    gpu_device: &ComputeDevice,
    engine: &dyn GemmEngine,
    cpu_budget: usize,
) -> Result<()> {
    clear_gpu_crown_working_set(gpu_device)?;
    let seconds = measure_seconds(|| case.run_production_from_fresh_ibp(engine))?;
    write_measured_phase(out, case, "wgpu_production_from_ibp", seconds, cpu_budget)
}

fn measure_production_phase(
    out: &mut impl Write,
    case: &BenchCase,
    gpu_device: &ComputeDevice,
    engine: &dyn GemmEngine,
    cpu_budget: usize,
) -> Result<()> {
    clear_gpu_crown_working_set(gpu_device)?;
    let seconds = measure_seconds(|| case.run_gpu_production(engine))?;
    write_measured_phase(out, case, "wgpu_production_cold", seconds, cpu_budget)
}

/// Measure GPU CROWN with a warm plan cache (no clear between calls).
///
/// This must run immediately after `measure_production_phase` so the plan
/// cache is already populated. The difference between cold and warm is the
/// plan creation overhead that the cache eliminates on subsequent calls.
fn measure_production_warm_phase(
    out: &mut impl Write,
    case: &BenchCase,
    engine: &dyn GemmEngine,
    cpu_budget: usize,
) -> Result<()> {
    let seconds = measure_seconds(|| case.run_gpu_production(engine))?;
    write_measured_phase(out, case, "wgpu_production_warm", seconds, cpu_budget)
}

fn measure_profiled_warm_phase(
    out: &mut impl Write,
    case: &BenchCase,
    gpu_device: &ComputeDevice,
    cpu_budget: usize,
    profile_gpu: bool,
    profile_host: bool,
) -> Result<()> {
    if !(profile_gpu || profile_host) {
        return Ok(());
    }
    let ComputeDevice::Wgpu(device) = gpu_device else {
        return Ok(());
    };
    let gpu_timestamp_supported = device.supports_timestamp_queries();
    if profile_gpu && !gpu_timestamp_supported && !profile_host {
        return write_unsupported_profile_phase(out, case, cpu_budget);
    }

    if profile_gpu && gpu_timestamp_supported {
        device.set_crown_timestamp_profiling(true)?;
    }
    if profile_host {
        device.set_crown_host_timing_profiling(true)?;
    }
    let profile_result = (|| {
        if profile_gpu && gpu_timestamp_supported {
            let _ = device.take_last_crown_timestamp_profile()?;
        }
        if profile_host {
            let _ = device.take_last_crown_host_timing_profile()?;
        }
        case.run_gpu_production(device.as_ref())?;
        let gpu_profile = if profile_gpu && gpu_timestamp_supported {
            Some(device.take_last_crown_timestamp_profile()?.ok_or_else(|| {
                NyError::InternalError(format!(
                    "{}: missing timestamp profile after profiled warm CROWN run",
                    case.name()
                ))
            })?)
        } else {
            None
        };
        let host_profile = if profile_host {
            Some(
                device
                    .take_last_crown_host_timing_profile()?
                    .ok_or_else(|| {
                        NyError::InternalError(format!(
                            "{}: missing host timing profile after profiled warm CROWN run",
                            case.name()
                        ))
                    })?,
            )
        } else {
            None
        };
        Ok((gpu_profile, host_profile))
    })();
    let disable_host_result = if profile_host {
        device.set_crown_host_timing_profiling(false)
    } else {
        Ok(())
    };
    let disable_gpu_result = if profile_gpu && gpu_timestamp_supported {
        device.set_crown_timestamp_profiling(false)
    } else {
        Ok(())
    };
    disable_host_result?;
    disable_gpu_result?;

    let (gpu_profile, host_profile) = profile_result?;
    if profile_gpu {
        if let Some(profile) = gpu_profile.as_ref() {
            write_gpu_profile_rows(out, case, cpu_budget, profile)?;
        } else {
            write_unsupported_profile_phase(out, case, cpu_budget)?;
        }
    }
    if let Some(profile) = host_profile.as_ref() {
        write_host_profile_rows(out, case, cpu_budget, profile)?;
    }
    Ok(())
}

fn write_unsupported_profile_phase(
    out: &mut impl Write,
    case: &BenchCase,
    cpu_budget: usize,
) -> Result<()> {
    write_csv_row(
        out,
        &MeasurementRow::skipped(
            case,
            "wgpu_production_profile_total",
            cpu_budget,
            "timestamp_queries_unsupported",
        ),
    )
}

/// Graph path measurement phases (#3716): convert the sequential workload to
/// `GraphNetwork` and measure the CROWN-IBP collection path that real VNN-COMP
/// ONNX models exercise through `verify_graph`.
///
/// With `--graph`: IBP forward + CROWN-IBP collection (CPU and engine).
/// With `--graph-full --case <name>`: also includes the full graph CROWN
/// backward which is extremely slow (43+ min for metaroom) and not the
/// VNN-COMP code path.
fn measure_graph_phases(
    out: &mut impl Write,
    case: &BenchCase,
    engine: &dyn GemmEngine,
    cpu_budget: usize,
    options: GraphMeasurementOptions,
) -> Result<()> {
    let ibp = measure_seconds(|| case.run_graph_ibp())?;
    write_measured_phase(out, case, "graph_ibp_forward", ibp, cpu_budget)?;
    if options.include_cpu_collection {
        let cpu = measure_seconds(|| case.run_graph_crown_ibp_collection(None))?;
        write_measured_phase(out, case, "graph_crown_ibp_collection_cpu", cpu, cpu_budget)?;
    }
    let eng = measure_seconds(|| case.run_graph_crown_ibp_collection(Some(engine)))?;
    write_measured_phase(
        out,
        case,
        "graph_crown_ibp_collection_engine",
        eng,
        cpu_budget,
    )?;
    if options.include_full_crown {
        let crown = measure_seconds(|| case.run_graph_crown(Some(engine)))?;
        write_measured_phase(out, case, "graph_crown_with_engine", crown, cpu_budget)?;
    }
    Ok(())
}

fn measure_case(
    out: &mut impl Write,
    case: &BenchCase,
    gpu_device: &ComputeDevice,
    engine: &dyn GemmEngine,
    cpu_budget: usize,
    args: &MeasurementArgs,
) -> Result<()> {
    if args.production_only() {
        measure_production_phase(out, case, gpu_device, engine, cpu_budget)?;
        measure_production_warm_phase(out, case, engine, cpu_budget)?;
        measure_profiled_warm_phase(
            out,
            case,
            gpu_device,
            cpu_budget,
            args.profile_gpu(),
            args.profile_host(),
        )?;
        return Ok(());
    }

    measure_cpu_phase(out, case, gpu_device, engine, cpu_budget)?;
    measure_ibp_phase(out, case, cpu_budget)?;
    measure_crown_ibp_phase(out, case, gpu_device, engine, cpu_budget)?;
    measure_production_from_ibp_phase(out, case, gpu_device, engine, cpu_budget)?;
    measure_production_phase(out, case, gpu_device, engine, cpu_budget)?;
    measure_production_warm_phase(out, case, engine, cpu_budget)?;
    measure_profiled_warm_phase(
        out,
        case,
        gpu_device,
        cpu_budget,
        args.profile_gpu(),
        args.profile_host(),
    )?;
    // GraphNetwork phases (#3716): gated behind --graph.
    // --graph: fast phases (IBP + CROWN-IBP collection) matching VNN-COMP path.
    // --graph-full: case-scoped opt-in for the extremely slow full graph CROWN
    // backward.
    if args.graph() {
        measure_graph_phases(
            out,
            case,
            engine,
            cpu_budget,
            GraphMeasurementOptions {
                include_cpu_collection: !args.graph_engine_only(),
                include_full_crown: args.graph_full(),
            },
        )?;
    }
    Ok(())
}

fn sanitize_detail(detail: &str) -> String {
    detail
        .chars()
        .map(|ch| match ch {
            ',' | '\n' | '\r' => ' ',
            _ => ch,
        })
        .collect()
}

fn panic_detail(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return sanitize_detail(message);
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return sanitize_detail(message);
    }
    "panic_without_string_payload".to_string()
}

fn validate_case_filter(args: &MeasurementArgs, cases: &[BenchCase]) -> Result<()> {
    if let Some(filter) = args.case_filter() {
        if !cases.iter().any(|case| case.name() == filter) {
            let supported_cases = cases
                .iter()
                .map(BenchCase::name)
                .collect::<Vec<_>>()
                .join(", ");
            return Err(NyError::InvalidSpec(format!(
                "unknown benchmark case `{filter}`; supported cases: {supported_cases}"
            )));
        }
    }
    Ok(())
}

fn run_with_output_for_cases(
    out: &mut impl Write,
    args: &MeasurementArgs,
    cases: &[BenchCase],
    cpu_budget: usize,
) -> Result<()> {
    let mut failures = Vec::new();

    validate_case_filter(args, cases)?;
    write_csv_header(out)?;
    for case in cases {
        if args
            .case_filter()
            .is_some_and(|filter| filter != case.name())
        {
            continue;
        }

        let gpu_device = match ComputeDevice::new(Backend::Wgpu) {
            Ok(device) => device,
            Err(err) => {
                let detail = sanitize_detail(&format!("wgpu_init_failed: {err}"));
                write_failed_phase(out, case, "device_init", cpu_budget, &detail)?;
                failures.push(format!("{}: {detail}", case.name()));
                continue;
            }
        };
        let engine: &dyn GemmEngine = &gpu_device;

        match panic::catch_unwind(AssertUnwindSafe(|| {
            measure_case(out, case, &gpu_device, engine, cpu_budget, args)
        })) {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                let detail = sanitize_detail(&err.to_string());
                write_failed_phase(out, case, "case_error", cpu_budget, &detail)?;
                failures.push(format!("{}: {detail}", case.name()));
            }
            Err(payload) => {
                let detail = panic_detail(payload);
                write_failed_phase(out, case, "case_panic", cpu_budget, &detail)?;
                failures.push(format!("{}: {detail}", case.name()));
            }
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(NyError::InternalError(format!(
            "measurement failures: {}",
            failures.join("; ")
        )))
    }
}

fn run_with_output(out: &mut impl Write, args: &MeasurementArgs) -> Result<()> {
    let cpu_budget = cpu_crown_dense_budget_bytes();
    let cases = build_bench_cases()?;
    run_with_output_for_cases(out, args, &cases, cpu_budget)
}

fn run() -> Result<()> {
    let args = MeasurementArgs::parse_from(std::env::args().skip(1))?;
    if let Some(output_path) = args.output_path() {
        let mut out = create_output_file(output_path)?;
        run_with_output(&mut out, &args)?;
        out.flush().map_err(|err| {
            NyError::InternalError(format!(
                "failed to flush output file `{}`: {err}",
                output_path.display()
            ))
        })?;
        return Ok(());
    }

    let stdout = io::stdout();
    let mut out = stdout.lock();
    run_with_output(&mut out, &args)
}

fn main() {
    // Enable per-node CROWN-IBP timing instrumentation via RUST_LOG=ny_propagate=info
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(io::stderr)
        .init();

    if let Err(err) = run() {
        let _ = writeln!(io::stderr().lock(), "{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
#[path = "measure_crown_backward_workloads/tests.rs"]
mod tests;
