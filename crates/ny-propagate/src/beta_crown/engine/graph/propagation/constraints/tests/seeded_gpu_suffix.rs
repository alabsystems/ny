// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use ny_core::{
    GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownResult, GpuCrownSeed, NaiveCpuGemmEngine,
};
use ny_test_utils::assert_bounded_tensor_close;

use crate::beta_crown::{GraphCrownContext, GraphNeuronConstraint, GraphSplitHistory};
use crate::BoundedTensor;

use super::super::backward::BackwardMode;
use super::super::patches::ConstrainedPatchesPolicy;
use super::patches::{
    assert_storing_intermediate_capture_3813, build_two_conv_relu_graph_3813,
    build_two_conv_relu_input_3813, no_deadline_verifier, run_constrained_backward_with_policy,
    storing_intermediates_mode_3813,
};
use super::support::assert_cache_bounds_close;

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

#[derive(Clone)]
struct SeededGpuSuffixEngine {
    expected_lower: Vec<f32>,
    expected_upper: Vec<f32>,
    expected_num_specs: usize,
    seeded_calls: Arc<AtomicUsize>,
    legacy_calls: Arc<AtomicUsize>,
}

impl SeededGpuSuffixEngine {
    fn from_expected(expected: &BoundedTensor) -> Self {
        Self {
            expected_lower: expected.lower().iter().copied().collect(),
            expected_upper: expected.upper().iter().copied().collect(),
            expected_num_specs: expected.len(),
            seeded_calls: Arc::new(AtomicUsize::new(0)),
            legacy_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn malformed(expected_num_specs: usize, lower: Vec<f32>, upper: Vec<f32>) -> Self {
        Self {
            expected_lower: lower,
            expected_upper: upper,
            expected_num_specs,
            seeded_calls: Arc::new(AtomicUsize::new(0)),
            legacy_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn seeded_calls(&self) -> usize {
        self.seeded_calls.load(Ordering::SeqCst)
    }

    fn legacy_calls(&self) -> usize {
        self.legacy_calls.load(Ordering::SeqCst)
    }
}

impl GemmEngine for SeededGpuSuffixEngine {
    fn gemm_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }

    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}

impl GpuCrownBackward for SeededGpuSuffixEngine {
    fn crown_backward_gpu(
        &self,
        _layers: &[GpuCrownLayer],
        _spec: &[f32],
        _num_specs: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> ny_core::Result<GpuCrownResult> {
        self.legacy_calls.fetch_add(1, Ordering::SeqCst);
        panic!("legacy GPU CROWN entrypoint should not be used in constrained suffix tests")
    }

    fn crown_backward_gpu_seeded(
        &self,
        layers: &[GpuCrownLayer],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> ny_core::Result<GpuCrownResult> {
        self.seeded_calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(
            gpu_layer_kinds(layers),
            vec!["Conv2d", "Activation", "Conv2d"],
            "constrained suffix should hand off the remaining Conv2d trunk"
        );
        assert_eq!(
            seed.num_specs, self.expected_num_specs,
            "seeded suffix should preserve the live objective row count"
        );
        assert_eq!(
            seed.current_dim, 4,
            "seeded suffix should start at the 1x2x2 relu2/conv2 carrier"
        );
        assert_eq!(
            seed.lower_a.len(),
            seed.num_specs * seed.current_dim,
            "seeded lower A must stay dense row-major"
        );
        assert_eq!(
            seed.upper_a.len(),
            seed.num_specs * seed.current_dim,
            "seeded upper A must stay dense row-major"
        );
        assert_eq!(
            input_lower.len(),
            input_upper.len(),
            "seeded GPU suffix input bounds must stay aligned"
        );
        Ok(GpuCrownResult {
            lower_bounds: self.expected_lower.clone(),
            upper_bounds: self.expected_upper.clone(),
        })
    }
}

#[test]
fn test_constrained_seeded_gpu_suffix_matches_cpu_baseline_3813() {
    // Fast-mock dispatch test — gate OFF so the fast seeded GPU suffix runs (the
    // production default is now sound, which would mask this non-sound mock).
    // #gpu-crown-sound-default.
    let _gate = crate::sound_gpu_gate::test_lock::lock_gate();
    let verifier = no_deadline_verifier();
    let graph = build_two_conv_relu_graph_3813();
    let input = build_two_conv_relu_input_3813();
    let history = GraphSplitHistory::new();

    let baseline_context = GraphCrownContext::for_history(&history);
    let (baseline_result, baseline_cache) = run_constrained_backward_with_policy(
        &verifier,
        &graph,
        &input,
        &baseline_context,
        None,
        None,
        BackwardMode::Standard,
        ConstrainedPatchesPolicy::dense_only(),
    );

    let engine = SeededGpuSuffixEngine::from_expected(&baseline_result.output_bounds);
    let gpu_context = GraphCrownContext::for_history_and_engine(&history, Some(&engine));
    let (gpu_result, gpu_cache) = run_constrained_backward_with_policy(
        &verifier,
        &graph,
        &input,
        &gpu_context,
        None,
        None,
        BackwardMode::Standard,
        ConstrainedPatchesPolicy::dense_only(),
    );

    assert_bounded_tensor_close(
        &gpu_result.output_bounds,
        &baseline_result.output_bounds,
        1e-5,
        "constrained seeded GPU suffix parity",
    );
    assert_cache_bounds_close(
        &gpu_cache,
        &baseline_cache,
        "constrained seeded GPU suffix cache",
    );
    assert_eq!(
        engine.seeded_calls(),
        1,
        "seeded GPU suffix should trigger exactly once on the final unary trunk"
    );
    assert_eq!(
        engine.legacy_calls(),
        0,
        "constrained suffix should not route through the legacy identity GPU entrypoint"
    );
}

#[test]
fn constrained_seeded_gpu_suffix_malformed_payloads_restore_cpu_backward() {
    let _gate = crate::sound_gpu_gate::test_lock::lock_gate();
    let verifier = no_deadline_verifier();
    let graph = build_two_conv_relu_graph_3813();
    let input = build_two_conv_relu_input_3813();
    let history = GraphSplitHistory::new();
    let baseline_context = GraphCrownContext::for_history(&history);
    let (baseline_result, baseline_cache) = run_constrained_backward_with_policy(
        &verifier,
        &graph,
        &input,
        &baseline_context,
        None,
        None,
        BackwardMode::Standard,
        ConstrainedPatchesPolicy::dense_only(),
    );
    let rows = baseline_result.output_bounds.len();
    assert!(rows > 0);

    for (lower, upper, label) in [
        (vec![-1.0; rows.saturating_sub(1)], vec![1.0; rows], "shape"),
        (vec![f32::NAN; rows], vec![1.0; rows], "NaN"),
        (vec![-1.0; rows], vec![f32::INFINITY; rows], "infinity"),
        (vec![2.0; rows], vec![1.0; rows], "inversion"),
    ] {
        let engine = SeededGpuSuffixEngine::malformed(rows, lower, upper);
        let gpu_context = GraphCrownContext::for_history_and_engine(&history, Some(&engine));
        let (actual, cache) = run_constrained_backward_with_policy(
            &verifier,
            &graph,
            &input,
            &gpu_context,
            None,
            None,
            BackwardMode::Standard,
            ConstrainedPatchesPolicy::dense_only(),
        );
        assert_eq!(engine.seeded_calls(), 1, "{label}: GPU attempt count");
        assert_bounded_tensor_close(
            &actual.output_bounds,
            &baseline_result.output_bounds,
            1e-5,
            label,
        );
        assert_cache_bounds_close(&cache, &baseline_cache, label);
    }
}

#[test]
fn test_constrained_seeded_gpu_suffix_skips_storing_intermediates_3813() {
    let verifier = no_deadline_verifier();
    let graph = build_two_conv_relu_graph_3813();
    let input = build_two_conv_relu_input_3813();
    let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let baseline_context = GraphCrownContext::for_history(&history);
    let (baseline_result, baseline_cache) = run_constrained_backward_with_policy(
        &verifier,
        &graph,
        &input,
        &baseline_context,
        None,
        None,
        storing_intermediates_mode_3813(&graph, &history),
        ConstrainedPatchesPolicy::dense_only(),
    );

    let engine = SeededGpuSuffixEngine::from_expected(&baseline_result.output_bounds);
    let gpu_context = GraphCrownContext::for_history_and_engine(&history, Some(&engine));
    let (gpu_result, gpu_cache) = run_constrained_backward_with_policy(
        &verifier,
        &graph,
        &input,
        &gpu_context,
        None,
        None,
        storing_intermediates_mode_3813(&graph, &history),
        ConstrainedPatchesPolicy::dense_only(),
    );

    assert_bounded_tensor_close(
        &gpu_result.output_bounds,
        &baseline_result.output_bounds,
        1e-5,
        "storing-intermediates constrained seeded GPU suffix parity",
    );
    assert_cache_bounds_close(
        &gpu_cache,
        &baseline_cache,
        "storing-intermediates constrained seeded GPU suffix cache",
    );
    let intermediate = gpu_result
        .intermediate
        .expect("storing intermediates mode should still capture intermediate state");
    assert_storing_intermediate_capture_3813(&intermediate);
    assert_eq!(
        engine.seeded_calls(),
        0,
        "seeded GPU suffix must stay disabled in storing-intermediates mode"
    );
    assert_eq!(
        engine.legacy_calls(),
        0,
        "storing-intermediates mode should not call the legacy GPU entrypoint either"
    );
}
