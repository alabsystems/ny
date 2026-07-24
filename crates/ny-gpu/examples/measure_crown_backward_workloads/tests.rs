// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{
    measure_graph_phases, run_with_output, run_with_output_for_cases,
    write_unsupported_profile_phase, GraphMeasurementOptions,
};
use ny_core::NaiveCpuGemmEngine;
use ny_gpu::benchmark_support::crown_backward_cases::build_bench_cases;
use ny_gpu::benchmark_support::crown_backward_measurements::MeasurementArgs;

#[test]
fn test_write_unsupported_profile_phase_marks_profile_row_skipped() {
    let cases = build_bench_cases().expect("bench cases should build");
    let case = &cases[0];
    let mut out = Vec::new();

    write_unsupported_profile_phase(&mut out, case, 2048)
        .expect("unsupported timestamp-query rows should write cleanly");

    let csv = String::from_utf8(out).expect("csv output must be utf-8");
    assert!(csv.contains("acasxu_like,wgpu_production_profile_total,,"));
    assert!(csv.contains(",2048,skipped,timestamp_queries_unsupported"));
}

#[test]
fn test_measure_graph_phases_writes_fast_rows_without_full_crown() {
    let cases = build_bench_cases().expect("bench cases should build");
    let case = &cases[0];
    let mut out = Vec::new();

    measure_graph_phases(
        &mut out,
        case,
        &NaiveCpuGemmEngine,
        2048,
        GraphMeasurementOptions {
            include_cpu_collection: true,
            include_full_crown: false,
        },
    )
    .expect("graph measurement phases should succeed on the small benchmark case");

    let csv = String::from_utf8(out).expect("csv output must be utf-8");
    let lines: Vec<_> = csv.lines().collect();
    assert_eq!(
        lines.len(),
        3,
        "expected 3 fast graph phase rows (no full crown): {csv}"
    );
    for phase in [
        "graph_ibp_forward",
        "graph_crown_ibp_collection_cpu",
        "graph_crown_ibp_collection_engine",
    ] {
        let phase_line = lines
            .iter()
            .find(|line| line.contains(&format!("acasxu_like,{phase},")))
            .unwrap_or_else(|| panic!("missing graph measurement phase `{phase}` in csv: {csv}"));
        assert!(
            phase_line.contains(",measured,"),
            "expected graph phase `{phase}` to be recorded as measured: {csv}"
        );
    }
    assert!(
        !csv.contains("graph_crown_with_engine"),
        "full graph CROWN should not appear without graph_full=true: {csv}"
    );
}

#[test]
fn test_measure_graph_phases_engine_only_skips_cpu_row() {
    let cases = build_bench_cases().expect("bench cases should build");
    let case = &cases[0];
    let mut out = Vec::new();

    measure_graph_phases(
        &mut out,
        case,
        &NaiveCpuGemmEngine,
        2048,
        GraphMeasurementOptions {
            include_cpu_collection: false,
            include_full_crown: false,
        },
    )
    .expect("engine-only graph phases should succeed on the small benchmark case");

    let csv = String::from_utf8(out).expect("csv output must be utf-8");
    let lines: Vec<_> = csv.lines().collect();
    assert_eq!(
        lines.len(),
        2,
        "expected only graph IBP + engine rows in engine-only mode: {csv}"
    );
    assert!(
        csv.contains("acasxu_like,graph_ibp_forward,"),
        "engine-only mode should retain graph IBP row: {csv}"
    );
    assert!(
        csv.contains("acasxu_like,graph_crown_ibp_collection_engine,"),
        "engine-only mode should retain graph engine row: {csv}"
    );
    assert!(
        !csv.contains("graph_crown_ibp_collection_cpu"),
        "engine-only mode must skip the slow CPU graph collection row: {csv}"
    );
}

#[test]
fn test_run_with_output_graph_engine_only_filters_case_and_omits_slow_graph_rows() {
    let args = MeasurementArgs::parse_from(vec![
        "--case".to_string(),
        "acasxu_like".to_string(),
        "--graph-engine-only".to_string(),
    ])
    .expect("graph-engine-only benchmark args should parse");
    let mut out = Vec::new();

    run_with_output(&mut out, &args)
        .expect("graph-engine-only runner should succeed on the small benchmark case");

    let csv = String::from_utf8(out).expect("csv output must be utf-8");
    let lines: Vec<_> = csv.lines().collect();
    assert_eq!(
        lines.len(),
        9,
        "expected header plus 8 acasxu rows for graph-engine-only mode: {csv}"
    );

    let body = &lines[1..];
    assert!(
        body.iter().all(|line| line.starts_with("acasxu_like,")),
        "case filter should keep only acasxu rows: {csv}"
    );
    for phase in [
        "cpu_production",
        "ibp_forward",
        "wgpu_crown_ibp_from_ibp",
        "wgpu_production_from_ibp",
        "wgpu_production_cold",
        "wgpu_production_warm",
        "graph_ibp_forward",
        "graph_crown_ibp_collection_engine",
    ] {
        let phase_line = body
            .iter()
            .find(|line| line.contains(&format!("acasxu_like,{phase},")))
            .unwrap_or_else(|| panic!("missing graph-engine-only phase `{phase}` in csv: {csv}"));
        assert!(
            phase_line.contains(",measured,"),
            "expected graph-engine-only phase `{phase}` to be measured: {csv}"
        );
    }
    assert!(
        !csv.contains("graph_crown_ibp_collection_cpu"),
        "graph-engine-only runner must omit the CPU graph collection row: {csv}"
    );
    assert!(
        !csv.contains("graph_crown_with_engine"),
        "graph-engine-only runner must omit the full graph CROWN row: {csv}"
    );
    assert!(
        !csv.contains("soundnessbench_exact_like") && !csv.contains("metaroom_6cnn_ry_like"),
        "case filter should exclude the larger benchmark workloads: {csv}"
    );
}

#[test]
fn test_run_with_output_graph_engine_only_preserves_skipped_cpu_row_under_tight_budget() {
    let args = MeasurementArgs::parse_from(vec![
        "--case".to_string(),
        "acasxu_like".to_string(),
        "--graph-engine-only".to_string(),
    ])
    .expect("graph-engine-only benchmark args should parse");
    let cases = build_bench_cases().expect("bench cases should build");
    let mut out = Vec::new();

    run_with_output_for_cases(&mut out, &args, &cases, 1024)
        .expect("runner should preserve skipped CPU rows under a tight dense budget");

    let csv = String::from_utf8(out).expect("csv output must be utf-8");
    let lines: Vec<_> = csv.lines().collect();
    assert_eq!(
        lines.len(),
        9,
        "expected header plus 8 acasxu rows for graph-engine-only mode: {csv}"
    );

    let body = &lines[1..];
    let cpu_row = body
        .iter()
        .find(|line| line.starts_with("acasxu_like,cpu_production,"))
        .unwrap_or_else(|| panic!("missing skipped cpu_production row in csv: {csv}"));
    assert!(
        cpu_row.contains(",1024,skipped,dense_peak_exceeds_budget"),
        "tight-budget runner test should preserve skipped CPU row details: {csv}"
    );
    for phase in [
        "ibp_forward",
        "wgpu_crown_ibp_from_ibp",
        "wgpu_production_from_ibp",
        "wgpu_production_cold",
        "wgpu_production_warm",
        "graph_ibp_forward",
        "graph_crown_ibp_collection_engine",
    ] {
        let phase_line = body
            .iter()
            .find(|line| line.contains(&format!("acasxu_like,{phase},")))
            .unwrap_or_else(|| {
                panic!("missing tight-budget graph-engine-only phase `{phase}`: {csv}")
            });
        assert!(
            phase_line.contains(",measured,"),
            "expected tight-budget graph-engine-only phase `{phase}` to be measured: {csv}"
        );
    }
    assert!(
        !csv.contains("graph_crown_ibp_collection_cpu"),
        "tight-budget graph-engine-only runner must omit the CPU graph collection row: {csv}"
    );
    assert!(
        !csv.contains("graph_crown_with_engine"),
        "tight-budget graph-engine-only runner must omit the full graph CROWN row: {csv}"
    );
}

#[test]
fn test_measure_graph_phases_writes_all_rows_with_full_crown() {
    let cases = build_bench_cases().expect("bench cases should build");
    let case = &cases[0];
    let mut out = Vec::new();

    measure_graph_phases(
        &mut out,
        case,
        &NaiveCpuGemmEngine,
        2048,
        GraphMeasurementOptions {
            include_cpu_collection: true,
            include_full_crown: true,
        },
    )
    .expect("full graph measurement should succeed on the small benchmark case");

    let csv = String::from_utf8(out).expect("csv output must be utf-8");
    let lines: Vec<_> = csv.lines().collect();
    assert_eq!(
        lines.len(),
        4,
        "expected 4 graph phase rows (including full crown): {csv}"
    );
    for phase in [
        "graph_ibp_forward",
        "graph_crown_ibp_collection_cpu",
        "graph_crown_ibp_collection_engine",
        "graph_crown_with_engine",
    ] {
        let phase_line = lines
            .iter()
            .find(|line| line.contains(&format!("acasxu_like,{phase},")))
            .unwrap_or_else(|| panic!("missing graph measurement phase `{phase}` in csv: {csv}"));
        assert!(
            phase_line.contains(",measured,"),
            "expected graph phase `{phase}` to be recorded as measured: {csv}"
        );
    }
}

#[test]
fn test_run_with_output_rejects_unknown_case_filter_before_writing_csv() {
    let args = MeasurementArgs::parse_from(vec![
        "--case".to_string(),
        "not_a_real_case".to_string(),
        "--graph-full".to_string(),
    ])
    .expect("unknown case names should still parse as raw args");
    let mut out = Vec::new();

    let err = run_with_output(&mut out, &args)
        .expect_err("unknown case filters must be rejected before measurement");

    assert!(
        out.is_empty(),
        "unknown case must not emit a partial CSV header"
    );
    let message = err.to_string();
    assert!(
        message.contains("unknown benchmark case `not_a_real_case`"),
        "unexpected error: {message}"
    );
    assert!(
        message.contains("acasxu_like"),
        "supported case list should help users fix the typo: {message}"
    );
}
