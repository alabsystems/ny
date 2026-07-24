// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use ndarray::{arr1, arr2};
use ny_core::NaiveCpuGemmEngine;
use ny_gpu::benchmark_support::crown_backward_cases::build_bench_cases;
use ny_gpu::benchmark_support::crown_backward_measurements::write_measured_phase;
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::{GraphNetwork, Layer, Network};
use ny_tensor::BoundedTensor;
use ny_test_utils::assert_bounded_tensor_close;

fn measure_seconds<T>(f: impl FnOnce() -> ny_core::Result<T>) -> ny_core::Result<f64> {
    let start = Instant::now();
    let _ = f()?;
    Ok(start.elapsed().as_secs_f64())
}

#[test]
fn test_graph_benchmark_core_phases_emit_measured_rows_for_small_case() {
    let cases = build_bench_cases().expect("bench cases should build");
    let case = &cases[0];
    let mut out = Vec::new();
    let cpu_budget = 2048;
    let engine = NaiveCpuGemmEngine;

    let ibp = measure_seconds(|| case.run_graph_ibp()).expect("graph IBP should succeed");
    write_measured_phase(&mut out, case, "graph_ibp_forward", ibp, cpu_budget)
        .expect("graph IBP row should write");

    let cpu = measure_seconds(|| case.run_graph_crown_ibp_collection(None))
        .expect("graph CROWN-IBP collection should succeed without engine");
    write_measured_phase(
        &mut out,
        case,
        "graph_crown_ibp_collection_cpu",
        cpu,
        cpu_budget,
    )
    .expect("graph CROWN-IBP CPU row should write");

    let eng = measure_seconds(|| case.run_graph_crown_ibp_collection(Some(&engine)))
        .expect("graph CROWN-IBP collection should succeed with NaiveCpuGemmEngine");
    write_measured_phase(
        &mut out,
        case,
        "graph_crown_ibp_collection_engine",
        eng,
        cpu_budget,
    )
    .expect("graph CROWN-IBP engine row should write");

    let crown = measure_seconds(|| case.run_graph_crown(Some(&engine)))
        .expect("full graph CROWN should succeed on the small case");
    write_measured_phase(&mut out, case, "graph_crown_with_engine", crown, cpu_budget)
        .expect("full graph CROWN row should write");

    let csv = String::from_utf8(out).expect("csv output must be utf-8");
    let lines: Vec<_> = csv.lines().collect();
    assert_eq!(
        lines.len(),
        4,
        "expected one CSV row per graph phase: {csv}"
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
fn test_graph_crown_matches_sequential_crown_on_small_network() {
    let mut network = Network::new();
    let layer1 = LinearLayer::new(arr2(&[[0.5, -0.25], [1.0, 0.75]]), Some(arr1(&[0.1, -0.2])))
        .expect("layer1 shapes should be valid");
    let layer2 = LinearLayer::new(arr2(&[[0.8, -1.2]]), Some(arr1(&[0.05])))
        .expect("layer2 shapes should be valid");

    network.add_layer(Layer::Linear(layer1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(layer2));

    let input = BoundedTensor::new(arr1(&[-0.1, 0.2]).into_dyn(), arr1(&[0.3, 0.6]).into_dyn())
        .expect("input bounds should be valid");

    let engine = NaiveCpuGemmEngine;
    let sequential = network
        .propagate_crown_with_engine(&input, Some(&engine))
        .expect("sequential CROWN should succeed");
    let graph = GraphNetwork::from_sequential(&network)
        .expect("sequential network should convert to GraphNetwork");
    let graph_bounds = graph
        .propagate_crown_with_engine(&input, Some(&engine))
        .expect("graph CROWN should succeed");

    assert_bounded_tensor_close(&graph_bounds, &sequential, 1e-5, "small_graph_crown_parity");
}
