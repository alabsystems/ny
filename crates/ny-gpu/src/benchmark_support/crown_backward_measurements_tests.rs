// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::crown_backward_measurements::{
    create_output_file, measure_or_skip_cpu_phase, write_csv_header, write_csv_row,
    write_failed_phase, CpuPhaseOutcome, MeasurementArgs, MeasurementRow,
};
use crate::benchmark_support::crown_backward_cases::build_bench_cases;

#[test]
fn test_parse_args_graph_defaults_to_false() {
    let args =
        MeasurementArgs::parse_from(Vec::<String>::new()).expect("empty args should be valid");
    assert!(!args.graph(), "graph() should default to false");
    assert!(!args.graph_full(), "graph_full() should default to false");
}

#[test]
fn test_parse_args_graph_full_implies_graph() {
    let args = MeasurementArgs::parse_from(vec![
        "--case".to_string(),
        "metaroom_6cnn_ry_like".to_string(),
        "--graph-full".to_string(),
    ])
    .expect("--graph-full with an explicit case should parse");
    assert!(args.graph(), "--graph-full must imply --graph");
    assert!(args.graph_full(), "--graph-full should set graph_full()");
    assert_eq!(args.case_filter(), Some("metaroom_6cnn_ry_like"));
}

#[test]
fn test_parse_args_graph_without_full() {
    let args =
        MeasurementArgs::parse_from(vec!["--graph".to_string()]).expect("--graph should parse");
    assert!(args.graph(), "--graph should set graph()");
    assert!(
        !args.graph_engine_only(),
        "--graph alone should not set graph_engine_only()"
    );
    assert!(!args.graph_full(), "--graph alone must not set graph_full");
}

#[test]
fn test_parse_args_graph_engine_only_implies_graph() {
    let args = MeasurementArgs::parse_from(vec!["--graph-engine-only".to_string()])
        .expect("--graph-engine-only should parse");
    assert!(args.graph(), "--graph-engine-only must imply --graph");
    assert!(
        args.graph_engine_only(),
        "--graph-engine-only should set graph_engine_only()"
    );
    assert!(
        !args.graph_full(),
        "--graph-engine-only should not set graph_full()"
    );
}

#[test]
fn test_parse_args_rejects_graph_engine_only_graph_full_combination() {
    let err = MeasurementArgs::parse_from(vec![
        "--case".to_string(),
        "soundnessbench_exact_like".to_string(),
        "--graph-engine-only".to_string(),
        "--graph-full".to_string(),
    ])
    .expect_err("--graph-engine-only with --graph-full must be rejected");

    let message = err.to_string();
    assert!(
        message.contains("--graph-engine-only cannot be combined with --graph-full"),
        "unexpected error: {message}"
    );
}

#[test]
fn test_parse_args_rejects_production_only_graph_combination() {
    let args = MeasurementArgs::parse_from(vec![
        "--case".to_string(),
        "soundnessbench_exact_like".to_string(),
        "--output".to_string(),
        "reports/benchmarks/current.csv".to_string(),
        "--profile-gpu".to_string(),
        "--profile-host".to_string(),
        "--production-only".to_string(),
        "--graph".to_string(),
    ])
    .expect_err("production-only graph combination must be rejected");

    let message = args.to_string();
    assert!(
        message.contains("--production-only cannot be combined with --graph or --graph-full"),
        "unexpected error: {message}"
    );
}

#[test]
fn test_parse_args_rejects_graph_full_without_case() {
    let err = MeasurementArgs::parse_from(vec!["--graph-full".to_string()])
        .expect_err("--graph-full without --case must be rejected");

    let message = err.to_string();
    assert!(
        message.contains("--graph-full requires --case <name>"),
        "unexpected error: {message}"
    );
}

#[test]
fn test_parse_args_supports_production_only() {
    let args = MeasurementArgs::parse_from(vec![
        "--case".to_string(),
        "soundnessbench_exact_like".to_string(),
        "--output".to_string(),
        "reports/benchmarks/current.csv".to_string(),
        "--profile-gpu".to_string(),
        "--profile-host".to_string(),
        "--production-only".to_string(),
    ])
    .expect("argument parsing should succeed");

    assert_eq!(args.case_filter(), Some("soundnessbench_exact_like"));
    assert_eq!(
        args.output_path()
            .map(|path| path.to_string_lossy().into_owned()),
        Some("reports/benchmarks/current.csv".to_string())
    );
    assert!(args.profile_gpu(), "--profile-gpu should set profile_gpu()");
    assert!(
        args.profile_host(),
        "--profile-host should set profile_host()"
    );
    assert!(
        args.production_only(),
        "--production-only should set production_only()"
    );
    assert!(!args.graph(), "--production-only should not set graph()");
}

#[test]
fn test_parse_args_rejects_unknown_flag() {
    let err = MeasurementArgs::parse_from(vec!["--bogus".to_string()])
        .expect_err("unknown flags must be rejected");
    let message = err.to_string();
    assert!(
        message.contains("unsupported argument `--bogus`"),
        "unexpected error: {message}"
    );
}

#[test]
fn test_measure_or_skip_cpu_phase_writes_skip_row_without_measuring() {
    let cases = build_bench_cases().expect("bench cases should build");
    let case = &cases[1];
    let mut out = Vec::new();

    let outcome = measure_or_skip_cpu_phase(&mut out, case, 1024, || {
        panic!("skip path must not invoke the measurement closure")
    })
    .expect("skip path should still write a CSV row");

    assert_eq!(outcome, CpuPhaseOutcome::Skipped);
    let csv = String::from_utf8(out).expect("csv output must be utf-8");
    assert!(
        csv.contains("soundnessbench_exact_like,cpu_production,,"),
        "skip row should contain case and phase: {csv}"
    );
    assert!(
        csv.contains(",1024,skipped,dense_peak_exceeds_budget"),
        "skip row should contain budget and skip status: {csv}"
    );
}

#[test]
fn test_measure_or_skip_cpu_phase_writes_measured_row() {
    let cases = build_bench_cases().expect("bench cases should build");
    let case = &cases[0];
    let mut out = Vec::new();

    let outcome = measure_or_skip_cpu_phase(&mut out, case, usize::MAX, || Ok(1.25))
        .expect("measured path should write a CSV row");

    assert_eq!(outcome, CpuPhaseOutcome::Measured);
    let csv = String::from_utf8(out).expect("csv output must be utf-8");
    assert!(
        csv.contains("acasxu_like,cpu_production,1.250000,"),
        "measured row should contain case, phase, and seconds: {csv}"
    );
    assert!(
        csv.contains(",measured,"),
        "measured row should contain measured status: {csv}"
    );
}

#[test]
fn test_write_csv_header_and_row_include_status_and_budget() {
    let cases = build_bench_cases().expect("bench cases should build");
    let case = &cases[0];
    let mut out = Vec::new();

    write_csv_header(&mut out).expect("header write should succeed");
    write_csv_row(
        &mut out,
        &MeasurementRow::measured(case, "wgpu_production_cold", 0.5, 2048),
    )
    .expect("row write should succeed");

    let csv = String::from_utf8(out).expect("csv output must be utf-8");
    assert!(csv.starts_with(
        "case,phase,seconds,parameter_count,estimated_cpu_peak_bytes,cpu_dense_budget_bytes,status,detail\n"
    ), "csv should start with expected header: {csv}");
    assert!(
        csv.contains(",2048,measured,"),
        "row should contain budget and measured status: {csv}"
    );
}

#[test]
fn test_create_output_file_creates_parent_directories() {
    let temp = tempfile::tempdir().expect("tempdir should succeed");
    let path = temp.path().join("nested").join("timing.csv");
    let _file = create_output_file(&path).expect("output file creation should succeed");
    assert!(path.exists(), "expected `{}` to exist", path.display());
}

#[test]
fn test_write_failed_phase_marks_failed_status() {
    let cases = build_bench_cases().expect("bench cases should build");
    let case = &cases[2];
    let mut out = Vec::new();

    write_failed_phase(&mut out, case, "case_panic", 2048, "dispatch_group_limit")
        .expect("failed row write should succeed");

    let csv = String::from_utf8(out).expect("csv output must be utf-8");
    assert!(
        csv.contains("metaroom_6cnn_ry_like,case_panic,,"),
        "failed row should contain case and phase: {csv}"
    );
    assert!(
        csv.contains(",2048,failed,dispatch_group_limit"),
        "failed row should contain budget and failed status: {csv}"
    );
}
