// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Real-ONNX graph-path timing checks for the GPU CROWN backward regression surface.
//!
//! The synthetic `ny-gpu` workload suite guards representative metaroom and
//! soundnessbench shapes, while `gpu_crown.rs` covers the real sequential
//! `Network` fast path with precomputed IBP. This module closes the remaining
//! gap by timing the actual `GraphNetwork::collect_crown_ibp_bounds_dag...`
//! path on the shipped ONNX benchmarks those regressions are meant to model.

use super::*;
use ndarray::{ArrayD, IxDyn};
use ny_core::GemmEngine;
use ny_gpu::{Backend, ComputeDevice};
use ny_propagate::{
    types::{BoundsProvenance, CrownIbpFallbackEvent, CrownIbpFallbackReason},
    GraphNetwork,
};
use ny_tensor::BoundedTensor;
use ny_test_utils::workspace_root;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
struct RealGraphProbeSummary {
    elapsed: Duration,
    output_provenance: Option<BoundsProvenance>,
    fallback_events: Vec<CrownIbpFallbackEvent>,
}

impl RealGraphProbeSummary {
    fn fallback_count(&self) -> usize {
        self.fallback_events.len()
    }

    fn first_fallback_layer(&self) -> Option<usize> {
        self.fallback_events.first().map(|event| event.layer_index)
    }
}

fn benchmark_path(rel: &str) -> PathBuf {
    workspace_root().join(rel)
}

fn gpu_real_graph_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn try_wgpu_device(label: &str) -> Option<ComputeDevice> {
    match ComputeDevice::new(Backend::Wgpu) {
        Ok(device) => Some(device),
        Err(err) => {
            eprintln!("{label}: SKIP: wgpu device not available ({err})");
            None
        }
    }
}

fn model_input(model: &OnnxModel, eps: f32) -> (BoundedTensor, Vec<usize>) {
    let input_spec = model
        .network
        .inputs
        .first()
        .expect("model has no input spec");
    let shape: Vec<usize> = input_spec.shape[1..]
        .iter()
        .map(|&dim| if dim > 0 { dim as usize } else { 1 })
        .collect();
    let center = ArrayD::zeros(IxDyn(&shape));
    let input = BoundedTensor::from_epsilon(center, eps).expect("BoundedTensor from_epsilon");
    (input, shape)
}

fn concrete_center(shape: &[usize]) -> BoundedTensor {
    let center = ArrayD::zeros(IxDyn(shape));
    BoundedTensor::new(center.clone(), center).expect("concrete center input")
}

fn assert_output_tightens_or_matches_ibp(
    output_bounds: &BoundedTensor,
    ibp_output: &BoundedTensor,
    label: &str,
) {
    let crown_lower = output_bounds
        .lower()
        .as_slice()
        .expect("crown lower contiguous");
    let crown_upper = output_bounds
        .upper()
        .as_slice()
        .expect("crown upper contiguous");
    let ibp_lower = ibp_output.lower().as_slice().expect("ibp lower contiguous");
    let ibp_upper = ibp_output.upper().as_slice().expect("ibp upper contiguous");

    assert_eq!(
        crown_lower.len(),
        ibp_lower.len(),
        "{label}: output dim mismatch between graph CROWN-IBP and IBP"
    );

    for idx in 0..crown_lower.len() {
        assert!(
            crown_lower[idx].is_finite() && crown_upper[idx].is_finite(),
            "{label}: non-finite graph CROWN-IBP bounds at dim {idx}: [{}, {}]",
            crown_lower[idx],
            crown_upper[idx],
        );
        assert!(
            crown_lower[idx] <= crown_upper[idx] + 1e-4,
            "{label}: inverted graph CROWN-IBP bounds at dim {idx}: [{}, {}]",
            crown_lower[idx],
            crown_upper[idx],
        );
        // The CROWN lane discharges its f32 coefficient-error envelope into
        // the concretized bound (sound outward charge proportional to the
        // accumulated |A|-magnitude), which the plain-IBP lane does not pay.
        // On large-magnitude outputs (metaroom bounds up to ~3.9e3) the
        // measured charge reaches ~1.5e-3 relative, so dominance over IBP
        // holds only up to a relative slack; 2e-3 covers the measured charge
        // with headroom while staying far below any real tightening CROWN
        // provides.
        let lower_slack = 1e-4_f32 + 2e-3 * ibp_lower[idx].abs();
        let upper_slack = 1e-4_f32 + 2e-3 * ibp_upper[idx].abs();
        assert!(
            crown_lower[idx] >= ibp_lower[idx] - lower_slack,
            "{label}: graph CROWN-IBP lower[{idx}]={:.6} looser than graph IBP lower={:.6}",
            crown_lower[idx],
            ibp_lower[idx],
        );
        assert!(
            crown_upper[idx] <= ibp_upper[idx] + upper_slack,
            "{label}: graph CROWN-IBP upper[{idx}]={:.6} looser than graph IBP upper={:.6}",
            crown_upper[idx],
            ibp_upper[idx],
        );
    }
}

fn assert_concrete_center_inside(
    bounds: &BoundedTensor,
    concrete_output: &BoundedTensor,
    label: &str,
) {
    let lower = bounds.lower().as_slice().expect("lower contiguous");
    let upper = bounds.upper().as_slice().expect("upper contiguous");
    let concrete = concrete_output
        .lower()
        .as_slice()
        .expect("concrete lower contiguous");

    assert_eq!(
        lower.len(),
        concrete.len(),
        "{label}: concrete output dim mismatch"
    );

    for idx in 0..lower.len() {
        assert!(
            concrete[idx] >= lower[idx] - 1e-4 && concrete[idx] <= upper[idx] + 1e-4,
            "{label}: concrete center output[{idx}]={:.6} outside graph CROWN-IBP [{:.6}, {:.6}]",
            concrete[idx],
            lower[idx],
            upper[idx],
        );
    }
}

fn load_real_graph_case(
    model_path: &str,
    eps: f32,
    label: &str,
) -> Option<(ComputeDevice, GraphNetwork, BoundedTensor, Vec<usize>)> {
    let onnx_path = benchmark_path(model_path);
    if !onnx_path.exists() {
        eprintln!(
            "{label}: SKIP: benchmark data not available at {}",
            onnx_path.display()
        );
        return None;
    }

    let gpu_device = try_wgpu_device(label)?;
    let model =
        load_onnx(&onnx_path).unwrap_or_else(|e| panic!("{label}: failed to load model: {e}"));
    let graph = model
        .to_graph_network()
        .unwrap_or_else(|e| panic!("{label}: to_graph_network failed: {e}"));
    let (input, shape) = model_input(&model, eps);
    Some((gpu_device, graph, input, shape))
}

fn real_graph_crown_ibp_timing(
    model_path: &str,
    eps: f32,
    label: &str,
) -> Option<RealGraphProbeSummary> {
    let (gpu_device, graph, input, shape) = load_real_graph_case(model_path, eps, label)?;

    eprintln!(
        "{label}: graph nodes={}, output='{}', eps={eps}",
        graph.num_nodes(),
        graph.output_name()
    );

    let ibp_output = graph
        .propagate_ibp(&input)
        .unwrap_or_else(|e| panic!("{label}: graph IBP failed: {e}"));
    let concrete_output = graph
        .propagate_ibp(&concrete_center(&shape))
        .unwrap_or_else(|e| panic!("{label}: concrete graph IBP failed: {e}"));

    let engine: &dyn GemmEngine = &gpu_device;
    let start = Instant::now();
    let result = graph
        .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, Some(engine))
        .unwrap_or_else(|e| panic!("{label}: graph CROWN-IBP collection failed: {e}"));
    let elapsed = start.elapsed();

    let output_bounds = result.bounds.get(graph.output_name()).unwrap_or_else(|| {
        panic!(
            "{label}: missing output bounds for '{}'",
            graph.output_name()
        )
    });
    let output_provenance = result.provenance_for_node(graph.output_name());
    eprintln!(
        "{label}: graph CROWN-IBP collection={:.3}s, fallbacks={}, first_fallback_layer={:?}, output_provenance={output_provenance:?}",
        elapsed.as_secs_f64(),
        result.fallback_count(),
        result.first_fallback_layer(),
    );
    for event in &result.fallback_events {
        eprintln!(
            "{label}: fallback at topo_index={} layer_type={} reason={:?}",
            event.layer_index, event.layer_type, event.reason,
        );
    }

    assert_output_tightens_or_matches_ibp(output_bounds, &ibp_output, label);
    assert_concrete_center_inside(output_bounds, &concrete_output, label);

    Some(RealGraphProbeSummary {
        elapsed,
        output_provenance,
        fallback_events: result.fallback_events,
    })
}

fn assert_no_fallbacks(summary: &RealGraphProbeSummary, label: &str) {
    assert_eq!(
        summary.output_provenance,
        Some(BoundsProvenance::Crown),
        "{label}: expected output provenance Crown, got {:?}",
        summary.output_provenance
    );
    assert_eq!(
        summary.fallback_count(),
        0,
        "{label}: expected zero fallback events, got {:?}",
        summary.fallback_events
    );
}

fn assert_conv2d_patches_budget_fallbacks(summary: &RealGraphProbeSummary, label: &str) {
    assert_eq!(
        summary.output_provenance,
        Some(BoundsProvenance::Crown),
        "{label}: expected output provenance Crown despite fallback events, got {:?}",
        summary.output_provenance
    );
    assert!(
        summary.fallback_count() > 0,
        "{label}: expected Conv2d patches-budget fallback events"
    );
    assert_eq!(
        summary.first_fallback_layer(),
        Some(2),
        "{label}: expected first fallback at topo index 2 for the current metaroom graph surface"
    );
    assert!(
        summary
            .fallback_events
            .iter()
            .all(|event| event.layer_type == "Conv2d"),
        "{label}: expected only Conv2d fallback layers, got {:?}",
        summary.fallback_events
    );
    assert!(
        summary
            .fallback_events
            .iter()
            .all(|event| event.reason == CrownIbpFallbackReason::PatchesBudgetExceeded),
        "{label}: expected only PatchesBudgetExceeded fallback reasons, got {:?}",
        summary.fallback_events
    );
}

fn report_competition_budget(label: &str, elapsed: Duration) {
    let budget = Duration::from_mins(3);
    let relation = if elapsed < budget { "within" } else { "over" };
    eprintln!(
        "{label}: graph collector runtime {:.3}s is {relation} the 180s competition budget",
        elapsed.as_secs_f64(),
    );
}

fn assert_within_competition_budget(label: &str, elapsed: Duration) {
    let budget = Duration::from_mins(3);
    report_competition_budget(label, elapsed);
    // Wall-clock competition budgets are asserted only under `--release`:
    // debug wall-clock measures the build profile, not the collector (same
    // policy as the avoice wall-clock budget policy, see `tests::core::avoice`
    // module docs). The `report_competition_budget` line above still records
    // the debug measurement.
    if cfg!(debug_assertions) {
        return;
    }
    assert!(
        elapsed < budget,
        "{label}: graph CROWN-IBP collection took {:.3}s, exceeds 180s VNN-COMP budget",
        elapsed.as_secs_f64(),
    );
}

/// Real metaroom graph-path timing for the engine-threaded CROWN-IBP collector.
///
/// This is the actual `GraphNetwork` path exercised by the VNN-COMP graph
/// verifier, not the sequential precomputed-IBP fast path.
#[ntest::timeout(900000)]
#[test]
fn test_gpu_graph_crown_ibp_real_metaroom_3397() {
    let _gpu_lock = gpu_real_graph_test_lock()
        .lock()
        .expect("real graph GPU test lock poisoned");
    let summary = match real_graph_crown_ibp_timing(
        "benchmarks/vnncomp2025/benchmarks/metaroom_2023/onnx/6cnn_ry_0_0_no_custom_OP.onnx",
        0.00001,
        "metaroom/graph_crown_ibp",
    ) {
        Some(summary) => summary,
        None => return,
    };
    assert_conv2d_patches_budget_fallbacks(&summary, "metaroom/graph_crown_ibp");
    report_competition_budget("metaroom/graph_crown_ibp", summary.elapsed);
}

/// Real soundnessbench graph-path timing for the engine-threaded CROWN-IBP collector.
///
/// This complements the existing sequential real-model timing test by covering
/// the actual DAG collector that the regression suite models.
///
/// The 180s competition budget and the 900s watchdog are release-only (see
/// `assert_within_competition_budget`); debug runs keep the soundness and
/// no-fallback assertions and report the measured time.
#[cfg_attr(not(debug_assertions), ntest::timeout(900000))]
#[test]
fn test_gpu_graph_crown_ibp_real_soundnessbench_3397() {
    let _gpu_lock = gpu_real_graph_test_lock()
        .lock()
        .expect("real graph GPU test lock poisoned");
    let summary = match real_graph_crown_ibp_timing(
        "benchmarks/vnncomp2025/benchmarks/soundnessbench/onnx/model.onnx",
        0.001,
        "soundnessbench/graph_crown_ibp",
    ) {
        Some(summary) => summary,
        None => return,
    };
    assert_no_fallbacks(&summary, "soundnessbench/graph_crown_ibp");
    assert_within_competition_budget("soundnessbench/graph_crown_ibp", summary.elapsed);
}
