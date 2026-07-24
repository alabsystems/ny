// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{arr1, arr2};
use ny_core::{
    GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownResult, NaiveCpuGemmEngine, Result,
};
use ny_test_utils::assert_bounded_tensor_close;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

struct ScriptedGpuFastPathEngine {
    lower_bounds: Vec<f32>,
    upper_bounds: Vec<f32>,
    gpu_calls: AtomicUsize,
    observed_num_specs: Mutex<Option<usize>>,
    observed_layer_kinds: Mutex<Option<Vec<&'static str>>>,
}

impl ScriptedGpuFastPathEngine {
    fn new(lower_bounds: Vec<f32>, upper_bounds: Vec<f32>) -> Self {
        Self {
            lower_bounds,
            upper_bounds,
            gpu_calls: AtomicUsize::new(0),
            observed_num_specs: Mutex::new(None),
            observed_layer_kinds: Mutex::new(None),
        }
    }

    fn gpu_calls(&self) -> usize {
        self.gpu_calls.load(Ordering::SeqCst)
    }

    fn observed_num_specs(&self) -> Option<usize> {
        *self
            .observed_num_specs
            .lock()
            .expect("observed_num_specs mutex should not be poisoned")
    }

    fn observed_layer_kinds(&self) -> Option<Vec<&'static str>> {
        self.observed_layer_kinds
            .lock()
            .expect("observed_layer_kinds mutex should not be poisoned")
            .clone()
    }
}

impl GemmEngine for ScriptedGpuFastPathEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }

    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}

impl GpuCrownBackward for ScriptedGpuFastPathEngine {
    fn crown_backward_gpu(
        &self,
        layers: &[GpuCrownLayer],
        _spec: &[f32],
        num_specs: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        self.gpu_calls.fetch_add(1, Ordering::SeqCst);
        *self
            .observed_num_specs
            .lock()
            .expect("observed_num_specs mutex should not be poisoned") = Some(num_specs);
        *self
            .observed_layer_kinds
            .lock()
            .expect("observed_layer_kinds mutex should not be poisoned") =
            Some(layer_kinds(layers));

        assert_eq!(
            input_lower.len(),
            input_upper.len(),
            "scripted GPU engine expects matching input bound lengths"
        );
        assert_eq!(
            self.lower_bounds.len(),
            self.upper_bounds.len(),
            "scripted GPU engine expected bounds must have matching lengths"
        );
        assert_eq!(
            self.lower_bounds.len(),
            num_specs,
            "scripted GPU engine expects one scalar bound per output spec"
        );

        Ok(GpuCrownResult {
            lower_bounds: self.lower_bounds.clone(),
            upper_bounds: self.upper_bounds.clone(),
        })
    }
}

fn layer_kinds(layers: &[GpuCrownLayer]) -> Vec<&'static str> {
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

fn build_linear_relu_network() -> Result<(Network, BoundedTensor)> {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.4, -0.1], [0.2, 0.3], [-0.5, 0.7]]),
        Some(arr1(&[0.1, -0.2, 0.05])),
    )?));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(
        arr2(&[[0.6, -0.4, 0.3], [-0.2, 0.5, 0.1]]),
        Some(arr1(&[0.0, 0.15])),
    )?));

    let input = BoundedTensor::new(
        arr1(&[-1.0, -0.5]).into_dyn(),
        arr1(&[1.0, 0.75]).into_dyn(),
    )?;
    Ok((network, input))
}

#[test]
fn test_propagate_crown_gpu_fast_path_nan_result_falls_back_to_cpu_3757() -> Result<()> {
    // Exercises the FAST (unsound f32) GPU CROWN path, which the
    // process-global soundness gate masks by default — hold the shared gate
    // lock (it sets the gate OFF) instead of depending on a gate-flipping
    // test elsewhere having leaked an OFF state.
    let _gate = sound_gpu_gate::test_lock::lock_gate();
    let (network, input) = build_linear_relu_network()?;

    let cpu_engine = NaiveCpuGemmEngine;
    let layer_bounds = network.collect_crown_ibp_bounds_with_engine_and_deadline(
        &input,
        Some(&cpu_engine),
        None,
    )?;
    let expected = network.propagate_crown_with_layer_bounds_and_engine_and_deadline_and_limits(
        &input,
        &layer_bounds,
        Some(&cpu_engine),
        None,
        None,
    )?;
    let ibp = network.propagate_ibp(&input)?;
    let cpu_lower: Vec<f32> = expected.lower().iter().copied().collect();
    let cpu_upper: Vec<f32> = expected.upper().iter().copied().collect();
    let ibp_lower: Vec<f32> = ibp.lower().iter().copied().collect();
    let ibp_upper: Vec<f32> = ibp.upper().iter().copied().collect();

    let mut scripted_lower = cpu_lower.clone();
    let mut scripted_upper = cpu_upper.clone();
    let nan_output_index = 0;

    let mut corrupted_endpoint = None;
    for i in 0..cpu_lower.len() {
        if i != nan_output_index && cpu_lower[i] > ibp_lower[i] + 1e-5 {
            let midpoint = f32::midpoint(cpu_lower[i], ibp_lower[i]);
            scripted_lower[i] = midpoint;
            corrupted_endpoint = Some(format!("lower[{i}]"));
            break;
        }
        if cpu_upper[i] + 1e-5 < ibp_upper[i] {
            let midpoint = f32::midpoint(cpu_upper[i], ibp_upper[i]);
            scripted_upper[i] = midpoint;
            corrupted_endpoint = Some(format!("upper[{i}]"));
            break;
        }
    }

    let corrupted = corrupted_endpoint
        .expect("fixture should contain at least one endpoint where CROWN is tighter than IBP");
    scripted_lower[nan_output_index] = f32::NAN;

    let scripted = ScriptedGpuFastPathEngine::new(scripted_lower, scripted_upper);
    let actual = network.propagate_crown_with_layer_bounds_and_engine_and_deadline_and_limits(
        &input,
        &layer_bounds,
        Some(&scripted),
        None,
        None,
    )?;

    assert_eq!(
        scripted.gpu_calls(),
        1,
        "GPU fast-path should be attempted once"
    );
    assert_eq!(scripted.observed_num_specs(), Some(expected.len()));
    assert_eq!(
        scripted.observed_layer_kinds(),
        Some(vec!["Linear", "Activation", "Linear"]),
        "ReLU network should extract as Linear -> Activation -> Linear"
    );
    let label = format!("nan fallback gpu vs cpu (corrupted: {corrupted})");
    assert_bounded_tensor_close(&actual, &expected, 1e-6, &label);
    assert!(
        actual
            .lower()
            .iter()
            .chain(actual.upper().iter())
            .all(|v| !v.is_nan()),
        "CPU fallback should be NaN-free (corrupted: {corrupted})"
    );
    Ok(())
}
