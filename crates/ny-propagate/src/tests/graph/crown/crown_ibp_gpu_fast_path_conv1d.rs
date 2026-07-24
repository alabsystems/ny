// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conv1d graph-level GPU CROWN-IBP fast-path coverage (#4023 item 1).
//!
//! `crown_ibp_gpu_fast_path.rs` exercises the sequential-graph GPU fast path on
//! Linear-ReLU-Linear graphs. This module verifies the same path with Conv1d
//! graphs, confirming that the Conv1d → Conv2d GPU layer mapping (height=1
//! equivalence from `gpu_extraction.rs`) works end-to-end through graph
//! CROWN-IBP collection.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use ndarray::{arr1, ArrayD, IxDyn};
use ny_core::{
    GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownResult, NaiveCpuGemmEngine, Result,
};
use ny_test_utils::assert_bounded_tensor_close;

use crate::types::BoundsProvenance;
use crate::*;

/// Build a Conv1d → ReLU → Flatten → Linear graph for GPU fast-path testing.
///
/// The 4-layer chain meets the ≥3 layer minimum in `try_gpu_crown_partial_backward`
/// and produces GPU layer_kinds `["Linear", "Activation", "Conv2d"]` (Flatten skipped).
fn build_conv1d_relu_flatten_linear_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    // Conv1d: 1→2 channels, kernel_size=3, stride=1, pad=0, input_length=6
    // Output: [2, 4] (2 channels × 4 positions)
    let conv_kernel =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![0.5, -0.25, 0.75, -0.2, 0.4, 0.1])
            .expect("conv1d kernel shape should be valid");
    let conv_bias = ndarray::Array1::from_vec(vec![0.15, -0.05]);
    let conv =
        Conv1dLayer::with_input_length(conv_kernel, Some(conv_bias), 1, 0, 6).expect("conv1d");
    graph.add_node(GraphNode::from_input("conv1d", Layer::Conv1d(conv)));

    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv1d".into()],
    ));

    graph.add_node(GraphNode::new(
        "flatten",
        Layer::Flatten(FlattenLayer::flatten_all()),
        vec!["relu".into()],
    ));

    // Linear: 8→2 (flattened 2×4=8 inputs → 2 outputs)
    let linear = LinearLayer::new(
        ndarray::arr2(&[
            [0.2, -0.1, 0.05, 0.3, -0.25, 0.4, -0.2, 0.15],
            [-0.3, 0.25, 0.15, -0.2, 0.1, -0.35, 0.4, -0.1],
        ]),
        Some(arr1(&[0.05, -0.1])),
    )
    .expect("linear");
    graph.add_node(GraphNode::new(
        "linear",
        Layer::Linear(linear),
        vec!["flatten".into()],
    ));
    graph.set_output("linear");

    // Input: [1, 6] (1 channel × 6 positions)
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 6]), vec![-0.5, -0.25, 0.0, -0.1, -0.2, -0.3])
            .expect("input lower"),
        ArrayD::from_shape_vec(IxDyn(&[1, 6]), vec![0.75, 0.5, 0.4, 0.6, 0.8, 0.7])
            .expect("input upper"),
    )
    .expect("input");

    (graph, input)
}

// ---------------------------------------------------------------------------
// Scripted GPU engine (minimal, Conv1d-specific)
// ---------------------------------------------------------------------------

struct Conv1dScriptedGpuExpectation {
    num_specs: usize,
    expected_layer_kinds: Vec<&'static str>,
    result_lower: Vec<f32>,
    result_upper: Vec<f32>,
}

struct Conv1dScriptedGpuEngine {
    gpu_calls: AtomicUsize,
    expectations: Mutex<VecDeque<Conv1dScriptedGpuExpectation>>,
}

impl Conv1dScriptedGpuEngine {
    fn new(expectations: Vec<Conv1dScriptedGpuExpectation>) -> Self {
        Self {
            gpu_calls: AtomicUsize::new(0),
            expectations: Mutex::new(VecDeque::from(expectations)),
        }
    }

    fn gpu_calls(&self) -> usize {
        self.gpu_calls.load(Ordering::SeqCst)
    }

    fn assert_all_consumed(&self) {
        let remaining = self.expectations.lock().expect("mutex").len();
        assert_eq!(
            remaining, 0,
            "Conv1dScriptedGpuEngine has {remaining} unconsumed expectations"
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

impl GemmEngine for Conv1dScriptedGpuEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }

    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}

impl GpuCrownBackward for Conv1dScriptedGpuEngine {
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
            .expect("mutex")
            .pop_front()
            .unwrap_or_else(|| {
                panic!(
                    "Conv1dScriptedGpuEngine: unexpected GPU call #{} (queue empty); \
                     num_specs={num_specs}, layer_kinds={:?}",
                    self.gpu_calls(),
                    gpu_layer_kinds(layers),
                )
            });

        assert_eq!(
            num_specs, expectation.num_specs,
            "Conv1dScriptedGpuEngine: num_specs mismatch"
        );

        let actual_kinds = gpu_layer_kinds(layers);
        assert_eq!(
            actual_kinds, expectation.expected_layer_kinds,
            "Conv1dScriptedGpuEngine: layer_kinds mismatch — Conv1d should map to Conv2d"
        );

        // Validate spec is an identity-like matrix (num_specs × num_specs)
        assert_eq!(
            spec.len(),
            num_specs * num_specs,
            "Conv1dScriptedGpuEngine: spec length mismatch"
        );

        // Validate input bounds are non-empty
        assert!(
            !input_lower.is_empty() && !input_upper.is_empty(),
            "Conv1dScriptedGpuEngine: empty input bounds"
        );

        Ok(GpuCrownResult {
            lower_bounds: expectation.result_lower,
            upper_bounds: expectation.result_upper,
        })
    }
}

/// Widen baseline CROWN bounds slightly toward IBP bounds to produce distinguishable
/// GPU results that are still within the IBP envelope.
fn widen_toward_ibp(baseline: &BoundedTensor, ibp: &BoundedTensor) -> (Vec<f32>, Vec<f32>) {
    let bl = baseline
        .lower()
        .as_slice()
        .expect("baseline lower contiguous");
    let bu = baseline
        .upper()
        .as_slice()
        .expect("baseline upper contiguous");
    let il = ibp.lower().as_slice().expect("ibp lower contiguous");
    let iu = ibp.upper().as_slice().expect("ibp upper contiguous");

    let mut lower = Vec::with_capacity(bl.len());
    let mut upper = Vec::with_capacity(bu.len());
    let mut changed = false;

    for i in 0..bl.len() {
        let new_l = if il[i] < bl[i] {
            changed = true;
            f32::midpoint(il[i], bl[i])
        } else {
            bl[i]
        };
        let new_u = if bu[i] < iu[i] {
            changed = true;
            f32::midpoint(bu[i], iu[i])
        } else {
            bu[i]
        };
        lower.push(new_l);
        upper.push(new_u);
    }

    assert!(
        changed,
        "Conv1d GPU fast-path fixture: IBP should leave slack for widening"
    );

    (lower, upper)
}

/// Assert GPU output bounds match or tighten scripted expectations, and differ
/// from CPU baseline (proving GPU result wasn't silently discarded).
fn assert_gpu_output_matches_script(
    gpu_output: &BoundedTensor,
    baseline_output: &BoundedTensor,
    scripted_lower: &[f32],
    scripted_upper: &[f32],
) {
    let gpu_lower = gpu_output.lower().as_slice().expect("contiguous");
    let gpu_upper = gpu_output.upper().as_slice().expect("contiguous");

    for i in 0..gpu_lower.len() {
        assert!(
            (gpu_lower[i] - scripted_lower[i]).abs() <= 1e-6 || gpu_lower[i] >= scripted_lower[i],
            "Conv1d GPU output lower[{i}]: actual={}, scripted={}",
            gpu_lower[i],
            scripted_lower[i],
        );
        assert!(
            (gpu_upper[i] - scripted_upper[i]).abs() <= 1e-6 || gpu_upper[i] <= scripted_upper[i],
            "Conv1d GPU output upper[{i}]: actual={}, scripted={}",
            gpu_upper[i],
            scripted_upper[i],
        );
    }

    let any_different = gpu_lower
        .iter()
        .zip(baseline_output.lower().iter())
        .chain(gpu_upper.iter().zip(baseline_output.upper().iter()))
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        any_different,
        "#4023 regression: GPU bounds identical to CPU baseline — scripted result discarded"
    );
}

/// #4023 regression: a Conv1d graph must exercise the GPU CROWN-IBP fast path,
/// confirming that Conv1d → Conv2d GPU layer mapping (height=1 equivalence from
/// `gpu_extraction.rs`) flows through graph CROWN-IBP collection end-to-end.
///
/// The existing `test_crown_ibp_dag_sequential_graph_uses_gpu_partial_fast_path_3599`
/// only tests Linear-ReLU-Linear graphs. This test covers the Conv1d gap.
#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_dag_graph_conv1d_uses_gpu_partial_fast_path_4023() {
    // This test exercises the FAST (unsound f32) GPU CROWN partial backward,
    // which the process-global soundness gate masks by default — hold the
    // shared gate lock (it sets the gate OFF) instead of depending on another
    // gate test having leaked an OFF state (lock order: gate → env budget,
    // same as gpu_partial_oracle).
    let _gate = sound_gpu_gate::test_lock::lock_gate();
    tests::with_crown_dense_budget_mb("2048", || {
        let (graph, input) = build_conv1d_relu_flatten_linear_graph();

        let baseline = graph
            .collect_crown_ibp_bounds_dag_with_status(&input)
            .unwrap();
        let ibp_bounds = graph.collect_node_bounds(&input).unwrap();

        let baseline_output = baseline.bounds.get("linear").expect("baseline 'linear'");
        let ibp_output = ibp_bounds.get("linear").expect("IBP 'linear'");

        // Backward chain: Linear → Flatten(skip) → Activation(ReLU) → Conv2d(Conv1d)
        let (scripted_lower, scripted_upper) = widen_toward_ibp(baseline_output, ibp_output);
        let expectation = Conv1dScriptedGpuExpectation {
            num_specs: baseline_output.len(),
            expected_layer_kinds: vec!["Linear", "Activation", "Conv2d"],
            result_lower: scripted_lower.clone(),
            result_upper: scripted_upper.clone(),
        };
        let scripted = Conv1dScriptedGpuEngine::new(vec![expectation]);

        let with_gpu = graph
            .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, Some(&scripted))
            .unwrap();

        assert_eq!(
            scripted.gpu_calls(),
            1,
            "#4023: Conv1d graph CROWN-IBP should issue one GPU partial backward"
        );
        scripted.assert_all_consumed();

        for name in ["conv1d", "relu", "flatten"] {
            if let (Some(base), Some(gpu)) = (baseline.bounds.get(name), with_gpu.bounds.get(name))
            {
                assert_bounded_tensor_close(gpu, base, 1e-6, &format!("non-output '{name}'"));
            }
        }

        let gpu_output = with_gpu.bounds.get("linear").expect("GPU 'linear'");
        assert_gpu_output_matches_script(
            gpu_output,
            baseline_output,
            &scripted_lower,
            &scripted_upper,
        );

        assert_eq!(
            with_gpu.provenance_for_node("linear"),
            Some(BoundsProvenance::Crown),
            "Conv1d graph GPU fast-path output should have Crown provenance"
        );
    });
}
