// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::crown_backward_cases::build_bench_cases;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_core::{
    ConvTranspose2dParams, GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownResult,
    GpuCrownSeed, NaiveCpuGemmEngine,
};
use ny_propagate::layers::{Conv2dLayer, FlattenLayer, LinearLayer, ReLULayer};
use ny_propagate::{GraphNetwork, Layer, Network};
use ny_tensor::BoundedTensor;
use ny_test_utils::assert_bounded_tensor_close;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, MutexGuard,
};

fn require_verdict_device() -> Arc<crate::WgpuDevice> {
    match crate::WgpuDevice::new_for_verdict(crate::WgpuVerdictRequest::new()) {
        Ok(device) => Arc::new(device),
        Err(error) => panic!(
            "verdict-qualified GPU required but qualification failed: {error}. \
             Run this #3813 parity regression on a conformant GPU host."
        ),
    }
}

fn gpu_test_serial_guard() -> MutexGuard<'static, ()> {
    static GPU_TEST_MUTEX: Mutex<()> = Mutex::new(());
    GPU_TEST_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Clone)]
struct CountingWgpuEngine {
    device: Arc<crate::WgpuDevice>,
    /// Successful fused `conv_transpose_2d` (GEMM + col2im) GPU calls.
    fused_calls: Arc<AtomicUsize>,
    /// Successful, production-accepted fast GPU CROWN backward calls whose
    /// extracted layer list contains a Conv2d.
    /// The graph planner may route a GPU-extractable unary conv chain through
    /// the GPU suffix instead of per-node `conv_transpose_2d`; both are valid
    /// GPU conv acceleration paths for the #3813 regression.
    suffix_calls: Arc<AtomicUsize>,
    /// Successful, production-accepted sound GPU CROWN backward calls whose
    /// extracted layer list contains a Conv2d.
    sound_calls: Arc<AtomicUsize>,
    /// Successful, production-accepted sound seeded GPU CROWN backward calls
    /// whose extracted layer list contains a Conv2d.
    sound_seeded_calls: Arc<AtomicUsize>,
}

impl CountingWgpuEngine {
    fn new(device: Arc<crate::WgpuDevice>) -> Self {
        Self {
            device,
            fused_calls: Arc::new(AtomicUsize::new(0)),
            suffix_calls: Arc::new(AtomicUsize::new(0)),
            sound_calls: Arc::new(AtomicUsize::new(0)),
            sound_seeded_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn fused_calls(&self) -> usize {
        self.fused_calls.load(Ordering::SeqCst)
    }

    fn sound_crown_calls(&self) -> usize {
        self.sound_calls.load(Ordering::SeqCst) + self.sound_seeded_calls.load(Ordering::SeqCst)
    }

    fn production_accepts_conv_crown(layers: &[GpuCrownLayer], result: &GpuCrownResult) -> bool {
        layers
            .iter()
            .any(|layer| matches!(layer, GpuCrownLayer::Conv2d { .. }))
            && !result
                .lower_bounds
                .iter()
                .chain(result.upper_bounds.iter())
                .any(|value| value.is_nan())
    }

    /// Total successful GPU conv-acceleration calls: fused `conv_transpose_2d`
    /// plus fast or sound GPU CROWN suffix dispatches. Either path means the
    /// conv-bearing backward chain completed on the GPU rather than falling
    /// back to the CPU.
    fn gpu_conv_calls(&self) -> usize {
        self.fused_calls() + self.suffix_calls.load(Ordering::SeqCst) + self.sound_crown_calls()
    }
}

impl GemmEngine for CountingWgpuEngine {
    fn gemm_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        self.device.gemm_f32(m, k, n, a, b)
    }

    fn conv_transpose_2d(
        &self,
        a_reshaped: &[f32],
        weight_col: &[f32],
        params: &ConvTranspose2dParams,
    ) -> ny_core::Result<Vec<f32>> {
        let result = self
            .device
            .conv_transpose_2d(a_reshaped, weight_col, params)?;
        self.fused_calls.fetch_add(1, Ordering::SeqCst);
        Ok(result)
    }

    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        // Expose `self` so fast and sound GPU CROWN dispatches are counted,
        // then delegate the actual work to the device.
        self.device.as_gpu_crown_backward().map(|_| self as _)
    }
}

impl GpuCrownBackward for CountingWgpuEngine {
    fn honors_crown_backward_deadline(&self) -> bool {
        true
    }

    fn set_crown_backward_deadline(&self, deadline: Option<std::time::Instant>) {
        self.device
            .as_gpu_crown_backward()
            .expect("device exposes GpuCrownBackward")
            .set_crown_backward_deadline(deadline);
    }

    fn crown_backward_gpu(
        &self,
        layers: &[GpuCrownLayer],
        spec: &[f32],
        num_specs: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> ny_core::Result<GpuCrownResult> {
        let gpu = self
            .device
            .as_gpu_crown_backward()
            .expect("device exposes GpuCrownBackward");
        let result = gpu.crown_backward_gpu(layers, spec, num_specs, input_lower, input_upper)?;
        if Self::production_accepts_conv_crown(layers, &result) {
            self.suffix_calls.fetch_add(1, Ordering::SeqCst);
        }
        Ok(result)
    }

    fn crown_backward_gpu_seeded(
        &self,
        layers: &[GpuCrownLayer],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> ny_core::Result<GpuCrownResult> {
        let gpu = self
            .device
            .as_gpu_crown_backward()
            .expect("device exposes GpuCrownBackward");
        let result = gpu.crown_backward_gpu_seeded(layers, seed, input_lower, input_upper)?;
        if Self::production_accepts_conv_crown(layers, &result) {
            self.suffix_calls.fetch_add(1, Ordering::SeqCst);
        }
        Ok(result)
    }

    fn crown_backward_gpu_sound(
        &self,
        layers: &[GpuCrownLayer],
        spec: &[f32],
        num_specs: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> ny_core::Result<GpuCrownResult> {
        let gpu = self
            .device
            .as_gpu_crown_backward()
            .expect("device exposes GpuCrownBackward");
        let result =
            gpu.crown_backward_gpu_sound(layers, spec, num_specs, input_lower, input_upper)?;
        if Self::production_accepts_conv_crown(layers, &result) {
            self.sound_calls.fetch_add(1, Ordering::SeqCst);
        }
        Ok(result)
    }

    fn provides_sound_gpu_crown(&self) -> bool {
        self.device
            .as_gpu_crown_backward()
            .is_some_and(|gpu| gpu.provides_sound_gpu_crown())
    }

    fn crown_backward_gpu_seeded_sound(
        &self,
        layers: &[GpuCrownLayer],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> ny_core::Result<GpuCrownResult> {
        let gpu = self
            .device
            .as_gpu_crown_backward()
            .expect("device exposes GpuCrownBackward");
        let result = gpu.crown_backward_gpu_seeded_sound(layers, seed, input_lower, input_upper)?;
        if Self::production_accepts_conv_crown(layers, &result) {
            self.sound_seeded_calls.fetch_add(1, Ordering::SeqCst);
        }
        Ok(result)
    }
}

fn build_small_conv_graph_case() -> (GraphNetwork, BoundedTensor) {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.5_f32, -0.25, 0.75, 0.4]).unwrap();
    let conv = Conv2dLayer::with_input_shape(kernel, Some(arr1(&[0.1_f32])), (1, 1), (0, 0), 4, 4)
        .expect("small Conv2d test case should be valid");
    let linear = LinearLayer::new(
        arr2(&[
            [0.25_f32, -0.5, 0.75, 0.1, 0.0, 0.5, -0.2, 0.4, 0.3],
            [-0.4, 0.3, 0.2, -0.6, 0.5, -0.1, 0.7, -0.2, 0.15],
        ]),
        Some(arr1(&[0.05_f32, -0.1])),
    )
    .expect("small linear head should be valid");

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(linear));

    let graph = GraphNetwork::from_sequential(&network)
        .expect("small sequential conv graph should convert");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), -0.2_f32),
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.6_f32),
    )
    .expect("small conv input bounds should be valid");

    (graph, input)
}

fn assert_node_bounds_close(
    actual: &HashMap<String, BoundedTensor>,
    expected: &HashMap<String, BoundedTensor>,
    epsilon: f32,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "graph CROWN-IBP node count changed: actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (name, expected_bounds) in expected {
        let actual_bounds = actual
            .get(name)
            .unwrap_or_else(|| panic!("missing graph CROWN-IBP bounds for node `{name}`"));
        assert_bounded_tensor_close(actual_bounds, expected_bounds, epsilon, name);
    }
}

/// Evaluate the small conv graph's EXACT output at a single concrete input
/// point by propagating IBP on a degenerate box (`lower == upper == x`).
///
/// Every layer in this network (Conv2d, ReLU, Flatten, Linear) maps a point to
/// a point under IBP — affine layers are exact and ReLU is monotone — so the
/// returned interval is degenerate and equals `f(x)`. This gives a relaxation-
/// free oracle for the true network output, used as the concrete reference for
/// the soundness check (GPU CROWN lower bound must be <= true min output).
fn exact_output_at(graph: &GraphNetwork, point: &ArrayD<f32>) -> Vec<f32> {
    let degenerate =
        BoundedTensor::new(point.clone(), point.clone()).expect("degenerate point box is valid");
    let out = graph
        .propagate_ibp(&degenerate)
        .expect("IBP on a degenerate point box should succeed");
    // l == u for a point input; take the lower (== upper) as the exact value.
    out.lower().iter().copied().collect()
}

/// Dense-sample the input box and assert `bounds` soundly enclose every sampled
/// true output: `lower[j] <= f(x)[j] <= upper[j]` for all samples `x` and
/// output indices `j`. Samples all `2^16` box corners is infeasible, so we use
/// all-low / all-high / midpoint plus a deterministic LCG sweep of random
/// interior points — enough to catch an unsound (too-tight) bound on this
/// small network. `tol` absorbs f32 rounding in the forward evaluation.
fn assert_bounds_sound_by_sampling(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bounds: &BoundedTensor,
    tol: f32,
    label: &str,
) {
    let lower = input.lower();
    let upper = input.upper();
    let shape = lower.shape().to_vec();
    let n = lower.len();
    let bl: Vec<f32> = bounds.lower().iter().copied().collect();
    let bu: Vec<f32> = bounds.upper().iter().copied().collect();

    let check = |x: ArrayD<f32>| {
        let fx = exact_output_at(graph, &x);
        for (j, &v) in fx.iter().enumerate() {
            assert!(
                v >= bl[j] - tol,
                "{label}: UNSOUND lower bound — true output[{j}]={v} < lower[{j}]={} (tol={tol})",
                bl[j]
            );
            assert!(
                v <= bu[j] + tol,
                "{label}: UNSOUND upper bound — true output[{j}]={v} > upper[{j}]={} (tol={tol})",
                bu[j]
            );
        }
    };

    let lo_vec: Vec<f32> = lower.iter().copied().collect();
    let hi_vec: Vec<f32> = upper.iter().copied().collect();

    // Corners that exercise the extreme activations: all-low, all-high, midpoint.
    check(ArrayD::from_shape_vec(IxDyn(&shape), lo_vec.clone()).unwrap());
    check(ArrayD::from_shape_vec(IxDyn(&shape), hi_vec.clone()).unwrap());
    let mid: Vec<f32> = lo_vec
        .iter()
        .zip(hi_vec.iter())
        .map(|(&l, &u)| f32::midpoint(l, u))
        .collect();
    check(ArrayD::from_shape_vec(IxDyn(&shape), mid).unwrap());

    // Deterministic LCG sweep of interior points (no rand dependency).
    let mut state: u64 = 0x9E3779B97F4A7C15;
    for _ in 0..2000 {
        let mut x = Vec::with_capacity(n);
        for i in 0..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let frac = ((state >> 11) as f32) / ((1u64 << 53) as f32);
            x.push(lo_vec[i] + frac * (hi_vec[i] - lo_vec[i]));
        }
        check(ArrayD::from_shape_vec(IxDyn(&shape), x).unwrap());
    }
}

/// Per-node soundness: every collected node bound must enclose that node's
/// true (relaxation-free) value, sampled densely across the input box. The
/// degenerate-box IBP pass produces exact per-node values (`l == u`); for each
/// node we assert `lower <= value <= upper` element-wise.
fn assert_node_bounds_sound_by_sampling(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    tol: f32,
) {
    let lo_vec: Vec<f32> = input.lower().iter().copied().collect();
    let hi_vec: Vec<f32> = input.upper().iter().copied().collect();
    let shape = input.lower().shape().to_vec();
    let n = lo_vec.len();

    let check = |x: ArrayD<f32>| {
        let degenerate = BoundedTensor::new(x.clone(), x).expect("degenerate point box valid");
        let exact = graph
            .collect_node_bounds(&degenerate)
            .expect("degenerate-box IBP collection should succeed");
        for (name, bound) in node_bounds {
            let Some(exact_node) = exact.get(name) else {
                continue;
            };
            let bl: Vec<f32> = bound.lower().iter().copied().collect();
            let bu: Vec<f32> = bound.upper().iter().copied().collect();
            for (j, &v) in exact_node.lower().iter().enumerate() {
                assert!(
                    v >= bl[j] - tol,
                    "node `{name}`: UNSOUND lower — true[{j}]={v} < lower[{j}]={} (tol={tol})",
                    bl[j]
                );
                assert!(
                    v <= bu[j] + tol,
                    "node `{name}`: UNSOUND upper — true[{j}]={v} > upper[{j}]={} (tol={tol})",
                    bu[j]
                );
            }
        }
    };

    check(ArrayD::from_shape_vec(IxDyn(&shape), lo_vec.clone()).unwrap());
    check(ArrayD::from_shape_vec(IxDyn(&shape), hi_vec.clone()).unwrap());
    let mut state: u64 = 0xD1B54A32D192ED03;
    for _ in 0..1000 {
        let mut x = Vec::with_capacity(n);
        for i in 0..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let frac = ((state >> 11) as f32) / ((1u64 << 53) as f32);
            x.push(lo_vec[i] + frac * (hi_vec[i] - lo_vec[i]));
        }
        check(ArrayD::from_shape_vec(IxDyn(&shape), x).unwrap());
    }
}

#[test]
fn test_graph_crown_with_naive_engine_matches_sequential_bounds_for_small_case() {
    let cases = build_bench_cases().expect("bench cases should build");
    let case = &cases[0];

    case.assert_graph_matches_sequential(Some(&NaiveCpuGemmEngine), 1e-2)
        .expect("graph and sequential benchmark bounds should stay aligned");
}

#[test]
fn test_graph_crown_qualified_wgpu_resident_conv_runs_soundly() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_verdict_device();
    let engine = CountingWgpuEngine::new(device);
    assert!(
        engine.provides_sound_gpu_crown() && engine.as_gpu_crown_backward().is_some(),
        "gpu-tests requires an explicitly qualified WGPU authority ladder"
    );
    let (mut graph, input) = build_small_conv_graph_case();
    graph.set_use_patches_mode(false);

    let baseline_bounds = graph
        .propagate_crown_with_engine(&input, None)
        .expect("CPU graph CROWN should succeed on the small conv case");
    let graph_bounds = graph
        .propagate_crown_with_engine(&input, Some(&engine))
        .expect("wgpu graph CROWN should succeed on the small conv case");
    assert_bounded_tensor_close(
        &graph_bounds,
        &baseline_bounds,
        1e-2,
        "small_conv_graph_crown_wgpu_vs_cpu_graph",
    );
    // The reviewed resident route now carries its taint word through Conv2d,
    // so a qualified device must do non-vacuous sound GPU work. This assertion
    // deliberately excludes an accidental all-CPU result that merely matches
    // the baseline. Host-side Conv and segment-resident streams remain separate
    // typed refusals; this graph uses the admitted whole-resident fold.
    assert!(
        engine.sound_crown_calls() > 0,
        "qualified graph CROWN did not exercise the admitted resident Conv route"
    );
    assert!(
        engine.gpu_conv_calls() > 0,
        "qualified graph CROWN completed without any GPU Conv work"
    );
    // Soundness: the GPU bound must enclose the true network output everywhere
    // in the input box — never tighter than the concrete minimum/maximum.
    assert_bounds_sound_by_sampling(
        &graph,
        &input,
        &graph_bounds,
        1e-3,
        "small_conv_graph_crown_wgpu_soundness",
    );
}

#[test]
fn test_graph_crown_ibp_qualified_wgpu_resident_conv_runs_soundly() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_verdict_device();
    let engine = CountingWgpuEngine::new(device);
    assert!(
        engine.provides_sound_gpu_crown() && engine.as_gpu_crown_backward().is_some(),
        "gpu-tests requires an explicitly qualified WGPU authority ladder"
    );
    let (mut graph, input) = build_small_conv_graph_case();
    graph.set_use_patches_mode(false);
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("graph IBP bounds should succeed on the small conv case");

    let baseline = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_width_threshold(
            &input,
            ibp_bounds.clone(),
            None,
            0.0,
        )
        .expect("CPU graph CROWN-IBP collection should succeed on the small conv case");
    let with_engine = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_engine_and_width_threshold(
            &input,
            ibp_bounds,
            None,
            Some(&engine),
            0.0,
        )
        .expect("wgpu graph CROWN-IBP collection should succeed on the small conv case");
    assert_node_bounds_close(&with_engine.bounds, &baseline.bounds, 1e-2);
    // As above, equality with CPU is not enough: the qualified CROWN-IBP lane
    // must actually enter the reviewed resident Conv route.
    assert!(
        engine.sound_crown_calls() > 0,
        "qualified graph CROWN-IBP did not exercise the admitted resident Conv route"
    );
    assert!(
        engine.gpu_conv_calls() > 0,
        "qualified graph CROWN-IBP completed without any GPU Conv work"
    );
    // Soundness: every collected node bound must enclose the true node output
    // sampled densely across the input box.
    assert_node_bounds_sound_by_sampling(&graph, &input, &with_engine.bounds, 1e-3);
}
