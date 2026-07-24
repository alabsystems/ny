// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression coverage for the sequential-graph GPU CROWN-IBP fast path (#3599)
//! and fork-join DAG per-node GPU engine threading (#4023).

use ny_test_utils::assert_bounded_tensor_close;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use ndarray::{arr1, arr2};
use ny_core::{
    GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownResult, NaiveCpuGemmEngine, Result,
};

use crate::layers::binary_ops::AddLayer;
use crate::layers::linear::LinearLayer;
use crate::tests::crown::helpers::CountingGemmEngine;
use crate::types::BoundsProvenance;
use crate::*;

use super::crown_ibp_engine::build_two_linear_relu_graph;

enum ScriptedGraphGpuResult {
    Bounds { lower: Vec<f32>, upper: Vec<f32> },
}

struct ScriptedGraphGpuExpectation {
    num_specs: usize,
    layer_kinds: Vec<&'static str>,
    spec: Vec<f32>,
    input_lower: Vec<f32>,
    input_upper: Vec<f32>,
    result: ScriptedGraphGpuResult,
}

struct ScriptedGraphGpuEngine {
    gpu_calls: AtomicUsize,
    expectations: Mutex<VecDeque<ScriptedGraphGpuExpectation>>,
}

impl ScriptedGraphGpuEngine {
    fn new(expectations: Vec<ScriptedGraphGpuExpectation>) -> Self {
        Self {
            gpu_calls: AtomicUsize::new(0),
            expectations: Mutex::new(VecDeque::from(expectations)),
        }
    }

    fn gpu_calls(&self) -> usize {
        self.gpu_calls.load(Ordering::SeqCst)
    }

    fn assert_all_consumed(&self) {
        let remaining = self
            .expectations
            .lock()
            .expect("expectations mutex should not be poisoned")
            .len();
        assert_eq!(
            remaining, 0,
            "ScriptedGraphGpuEngine has {remaining} unconsumed expectations"
        );
    }
}

fn gpu_layer_kinds(layers: &[GpuCrownLayer]) -> Vec<&'static str> {
    layers
        .iter()
        .map(|layer| match layer {
            GpuCrownLayer::Linear { .. } => "Linear",
            GpuCrownLayer::Activation { .. } | GpuCrownLayer::ActivationReluDualAlpha { .. } => {
                "Activation"
            }
            GpuCrownLayer::MaxPool2d { .. } => "MaxPool2d",
            GpuCrownLayer::Conv2d { .. } => "Conv2d",
        })
        .collect()
}

fn dense_identity_spec(output_dim: usize) -> Vec<f32> {
    let mut spec = vec![0.0; output_dim * output_dim];
    for i in 0..output_dim {
        spec[i * output_dim + i] = 1.0;
    }
    spec
}

fn assert_bounds_match_slices(
    actual: &BoundedTensor,
    expected_lower: &[f32],
    expected_upper: &[f32],
    tol: f32,
    label: &str,
) {
    let actual_lower = actual
        .lower()
        .as_slice()
        .expect("graph GPU fast-path lower should be contiguous");
    let actual_upper = actual
        .upper()
        .as_slice()
        .expect("graph GPU fast-path upper should be contiguous");

    assert_eq!(
        actual_lower.len(),
        expected_lower.len(),
        "{label}: lower length mismatch"
    );
    assert_eq!(
        actual_upper.len(),
        expected_upper.len(),
        "{label}: upper length mismatch"
    );

    for i in 0..actual_lower.len() {
        assert!(
            (actual_lower[i] - expected_lower[i]).abs() <= tol,
            "{label}: lower mismatch at index {i}: actual={}, expected={}, diff={}",
            actual_lower[i],
            expected_lower[i],
            (actual_lower[i] - expected_lower[i]).abs()
        );
        assert!(
            (actual_upper[i] - expected_upper[i]).abs() <= tol,
            "{label}: upper mismatch at index {i}: actual={}, expected={}, diff={}",
            actual_upper[i],
            expected_upper[i],
            (actual_upper[i] - expected_upper[i]).abs()
        );
    }
}

fn widen_within_ibp(baseline: &BoundedTensor, ibp: &BoundedTensor) -> (Vec<f32>, Vec<f32>) {
    let baseline_lower = baseline
        .lower()
        .as_slice()
        .expect("baseline lower should be contiguous");
    let baseline_upper = baseline
        .upper()
        .as_slice()
        .expect("baseline upper should be contiguous");
    let ibp_lower = ibp
        .lower()
        .as_slice()
        .expect("IBP lower should be contiguous");
    let ibp_upper = ibp
        .upper()
        .as_slice()
        .expect("IBP upper should be contiguous");

    let mut widened_lower = Vec::with_capacity(baseline.len());
    let mut widened_upper = Vec::with_capacity(baseline.len());
    let mut changed = false;

    for i in 0..baseline.len() {
        let new_lower = if ibp_lower[i] < baseline_lower[i] {
            changed = true;
            f32::midpoint(ibp_lower[i], baseline_lower[i])
        } else {
            baseline_lower[i]
        };
        let new_upper = if baseline_upper[i] < ibp_upper[i] {
            changed = true;
            f32::midpoint(baseline_upper[i], ibp_upper[i])
        } else {
            baseline_upper[i]
        };
        widened_lower.push(new_lower);
        widened_upper.push(new_upper);
    }

    assert!(
        changed,
        "graph GPU fast-path fixture should leave at least one IBP-to-CROWN tightening slack"
    );

    (widened_lower, widened_upper)
}

fn node_bounds<'a>(
    bounds: &'a std::collections::HashMap<String, BoundedTensor>,
    name: &str,
) -> &'a BoundedTensor {
    bounds
        .get(name)
        .unwrap_or_else(|| panic!("expected bounds for node '{name}'"))
}

fn build_graph_fast_path_expectation(
    baseline_output: &BoundedTensor,
    ibp_output: &BoundedTensor,
    input: &BoundedTensor,
) -> (ScriptedGraphGpuExpectation, Vec<f32>, Vec<f32>) {
    let (scripted_lower, scripted_upper) = widen_within_ibp(baseline_output, ibp_output);
    let expectation = ScriptedGraphGpuExpectation {
        num_specs: baseline_output.len(),
        layer_kinds: vec!["Linear", "Activation", "Linear"],
        spec: dense_identity_spec(baseline_output.len()),
        input_lower: input.lower().iter().copied().collect(),
        input_upper: input.upper().iter().copied().collect(),
        result: ScriptedGraphGpuResult::Bounds {
            lower: scripted_lower.clone(),
            upper: scripted_upper.clone(),
        },
    };
    (expectation, scripted_lower, scripted_upper)
}

impl GemmEngine for ScriptedGraphGpuEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }

    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}

impl GpuCrownBackward for ScriptedGraphGpuEngine {
    fn crown_backward_gpu(
        &self,
        layers: &[GpuCrownLayer],
        spec: &[f32],
        num_specs: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        self.gpu_calls.fetch_add(1, Ordering::SeqCst);

        let expectation = self
            .expectations
            .lock()
            .expect("expectations mutex should not be poisoned")
            .pop_front()
            .unwrap_or_else(|| {
                panic!(
                    "ScriptedGraphGpuEngine: unexpected GPU call #{} (queue empty); \
                     num_specs={num_specs}, layer_kinds={:?}",
                    self.gpu_calls(),
                    gpu_layer_kinds(layers),
                )
            });

        assert_eq!(
            num_specs, expectation.num_specs,
            "ScriptedGraphGpuEngine: num_specs mismatch"
        );
        assert_eq!(
            gpu_layer_kinds(layers),
            expectation.layer_kinds,
            "ScriptedGraphGpuEngine: layer order mismatch"
        );
        assert_eq!(
            spec, expectation.spec,
            "ScriptedGraphGpuEngine: spec matrix mismatch"
        );
        assert_eq!(
            input_lower, expectation.input_lower,
            "ScriptedGraphGpuEngine: input lower mismatch"
        );
        assert_eq!(
            input_upper, expectation.input_upper,
            "ScriptedGraphGpuEngine: input upper mismatch"
        );

        match expectation.result {
            ScriptedGraphGpuResult::Bounds { lower, upper } => Ok(GpuCrownResult {
                lower_bounds: lower,
                upper_bounds: upper,
            }),
        }
    }
}

/// Verify structural soundness properties of GPU-returned bounds:
/// (1) lower ≤ upper, (2) within IBP envelope, (3) distinguishable from CPU baseline.
fn assert_gpu_bounds_sound(
    gpu: &BoundedTensor,
    ibp: &BoundedTensor,
    baseline: &BoundedTensor,
    label: &str,
) {
    for (i, (&lo, &hi)) in gpu.lower().iter().zip(gpu.upper().iter()).enumerate() {
        assert!(
            lo <= hi,
            "{label}: bound ordering violated at index {i}: lower={lo} > upper={hi}"
        );
    }
    for (i, ((&gl, &gu), (&il, &iu))) in gpu
        .lower()
        .iter()
        .zip(gpu.upper().iter())
        .zip(ibp.lower().iter().zip(ibp.upper().iter()))
        .enumerate()
    {
        assert!(
            gl >= il - 1e-6,
            "{label}: lower[{i}] ({gl}) below IBP lower ({il})"
        );
        assert!(
            gu <= iu + 1e-6,
            "{label}: upper[{i}] ({gu}) above IBP upper ({iu})"
        );
    }
    let any_wider = gpu
        .lower()
        .iter()
        .zip(gpu.upper().iter())
        .zip(baseline.lower().iter().zip(baseline.upper().iter()))
        .any(|((&gl, &gu), (&bl, &bu))| gl < bl - 1e-6 || gu > bu + 1e-6);
    assert!(
        any_wider,
        "{label}: GPU bounds identical to CPU baseline — scripted result was silently discarded"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_dag_sequential_graph_uses_gpu_partial_fast_path_3599() {
    // This test exercises the FAST (unsound f32) GPU CROWN partial backward,
    // which the process-global soundness gate masks by default — hold the
    // shared gate lock (it sets the gate OFF) instead of depending on another
    // gate test having leaked an OFF state (lock order: gate → env budget,
    // same as gpu_partial_oracle).
    let _gate = sound_gpu_gate::test_lock::lock_gate();
    tests::with_crown_dense_budget_mb("2048", || {
        let (graph, input) = build_two_linear_relu_graph();
        let baseline = graph
            .collect_crown_ibp_bounds_dag_with_status(&input)
            .unwrap();
        let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
        let (expectation, scripted_lower, scripted_upper) = build_graph_fast_path_expectation(
            node_bounds(&baseline.bounds, "l2"),
            node_bounds(&ibp_bounds, "l2"),
            &input,
        );
        let scripted = ScriptedGraphGpuEngine::new(vec![expectation]);

        let with_engine = graph
            .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, Some(&scripted))
            .unwrap();

        assert_eq!(
            scripted.gpu_calls(),
            1,
            "#3599 regression: sequential graph CROWN-IBP should issue one GPU partial backward"
        );
        scripted.assert_all_consumed();

        for name in ["l1", "relu"] {
            assert_bounded_tensor_close(
                node_bounds(&with_engine.bounds, name),
                node_bounds(&baseline.bounds, name),
                1e-6,
                "sequential graph GPU partial fast path non-output node",
            );
        }
        assert_bounds_match_slices(
            node_bounds(&with_engine.bounds, "l2"),
            &scripted_lower,
            &scripted_upper,
            1e-6,
            "sequential graph GPU partial fast path l2",
        );

        assert_gpu_bounds_sound(
            node_bounds(&with_engine.bounds, "l2"),
            node_bounds(&ibp_bounds, "l2"),
            node_bounds(&baseline.bounds, "l2"),
            "sequential graph GPU partial fast path l2",
        );

        for name in baseline.bounds.keys() {
            assert_eq!(
                baseline.provenance_for_node(name),
                with_engine.provenance_for_node(name),
                "#3599 regression: sequential graph GPU fast path changed provenance at node '{name}'"
            );
        }
        assert_eq!(
            with_engine.provenance_for_node("l2"),
            Some(BoundsProvenance::Crown),
            "output node should remain on the Crown provenance path"
        );
    });
}

// ---------------------------------------------------------------------------
// Fork-join DAG: per-node GPU engine threading (#4023 item 2)
// ---------------------------------------------------------------------------

/// Build a diamond DAG that is NOT a sequential chain:
///
/// ```text
/// Input[2]
///   |
/// Linear1(2→2)
///   |
/// ReLU1
///  / \
/// L2a  L2b      (fan-out: relu1 consumed by two different Linears)
///  \  /
///  Add           (merge: binary op)
///   |
/// ReLU2          (output)
/// ```
///
/// `is_sequential_graph()` returns false because relu1 has two consumers.
/// CROWN-IBP collection falls through to the per-node O(N) loop, which
/// threads `engine` into each `propagate_crown_to_node` call.
fn build_diamond_dag() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let b1 = arr1(&[0.1_f32, -0.1]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("valid diamond linear1")),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    // Branch A
    let w2a = arr2(&[[0.8_f32, -0.3], [-0.2, 0.6]]);
    graph.add_node(GraphNode::new(
        "linear2a",
        Layer::Linear(LinearLayer::new(w2a, None).expect("valid diamond linear2a")),
        vec!["relu1".to_string()],
    ));

    // Branch B
    let w2b = arr2(&[[-0.4_f32, 0.7], [0.5, -0.1]]);
    graph.add_node(GraphNode::new(
        "linear2b",
        Layer::Linear(LinearLayer::new(w2b, None).expect("valid diamond linear2b")),
        vec!["relu1".to_string()],
    ));

    // Merge
    graph.add_node(GraphNode::binary(
        "add",
        Layer::Add(AddLayer),
        "linear2a",
        "linear2b",
    ));

    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["add".to_string()],
    ));
    graph.set_output("relu2");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .expect("valid diamond input");

    (graph, input)
}

/// #4023 item 2: verify that a fork-join (diamond) DAG threads the GemmEngine
/// through per-node CROWN-IBP backward passes.
///
/// Unlike the sequential-graph test above, this DAG cannot use the
/// `try_collect_crown_ibp_bounds_via_sequential_network` fast path because
/// relu1 fans out to two consumers (linear2a, linear2b) and the Add node is
/// a binary op. The engine must still be dispatched via the per-node loop
/// in `collect_crown_ibp_bounds_core_inner`.
#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_dag_diamond_fork_join_threads_engine_4023() {
    tests::with_crown_dense_budget_mb("2048", || {
        let (graph, input) = build_diamond_dag();

        // Baseline: no engine (default faer CPU GEMM)
        let baseline = graph
            .collect_crown_ibp_bounds_dag_with_status(&input)
            .expect("#4023 baseline diamond CROWN-IBP should succeed");

        // With CountingGemmEngine: verify engine is dispatched
        let engine = CountingGemmEngine::new();
        let with_engine = graph
            .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, Some(&engine))
            .expect("#4023 diamond CROWN-IBP with engine should succeed");

        // (1) Engine was actually called — fork-join DAGs thread engine per-node
        let calls = engine.gemm_calls();
        assert!(
            calls > 0,
            "#4023 regression: diamond fork-join DAG CROWN-IBP should dispatch GemmEngine, \
             got 0 calls — per-node engine threading broken for non-sequential graphs"
        );

        // (2) Bounds parity: CountingGemmEngine delegates to NaiveCpuGemmEngine,
        //     so results must match the baseline (which uses the default faer backend).
        for name in baseline.bounds.keys() {
            assert_bounded_tensor_close(
                node_bounds(&baseline.bounds, name),
                node_bounds(&with_engine.bounds, name),
                1e-5,
                &format!("#4023 diamond fork-join DAG engine parity at node '{name}'"),
            );
        }

        // (3) Provenance preserved — engine threading must not alter provenance.
        for name in baseline.bounds.keys() {
            assert_eq!(
                baseline.provenance_for_node(name),
                with_engine.provenance_for_node(name),
                "#4023 regression: diamond fork-join DAG engine changed provenance at node '{name}'"
            );
        }

        // (4) Output node should have Crown provenance (not IBP fallback).
        assert_eq!(
            with_engine.provenance_for_node("relu2"),
            Some(BoundsProvenance::Crown),
            "#4023: diamond DAG output should have Crown provenance, not IBP fallback"
        );
    });
}
