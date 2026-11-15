// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::{LinearLayer, ReLULayer};
use std::sync::atomic::{AtomicUsize, Ordering};

fn cancellation_network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::from_flat(vec![1.0 / 512.0; 512], 1, 512, Some(vec![-1.0])).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::from_flat(vec![10_000.0], 1, 1, None).unwrap(),
    ));
    network
}

#[test]
fn sequential_point_linear_cancellation_before_relu_has_no_certificate_padding() {
    let network = cancellation_network();
    let input = BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[2, 512]), 1.0)).unwrap();
    for mode in [LeadingAxisMode::Plain, LeadingAxisMode::PreserveLeadingAxis] {
        let point = propagate_concrete_point(&network, &input, None, mode).unwrap();
        assert_eq!(point.shape(), &[2, 1]);
        assert!(point.lower().iter().all(|&v| v == 0.0));
        assert_eq!(point.lower(), point.upper());
    }
    let interval = propagate_ibp_sound(&network, &input).unwrap();
    assert!(interval.upper().iter().all(|&v| v > 0.0));
}

struct ResidentPointEngine {
    fail: bool,
    resident: AtomicUsize,
    gemms: AtomicUsize,
}
impl ny_core::GpuIbpForward for ResidentPointEngine {
    fn ibp_forward_gpu(
        &self,
        _: &[GpuIbpLayer],
        _: &[f32],
        _: &[f32],
        shape: &[usize],
    ) -> Result<ny_core::GpuIbpResult> {
        self.resident.fetch_add(1, Ordering::SeqCst);
        assert_eq!(shape, &[2, 512]);
        if self.fail {
            return Err(NyError::UnsupportedOp("injected resident refusal".into()));
        }
        Ok(ny_core::GpuIbpResult {
            lower_bounds: vec![0.0; 2],
            // A valid loose interval whose center differs from the actual
            // zero point. A point evaluator must not use this box center.
            upper_bounds: vec![2.0; 2],
            output_shape: vec![2, 1],
        })
    }
}
impl GemmEngine for ResidentPointEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.gemms.fetch_add(1, Ordering::SeqCst);
        let mut c = vec![0.0; m * n];
        for row in 0..m {
            for out in 0..n {
                for inner in 0..k {
                    c[row * n + out] += a[row * k + inner] * b[inner * n + out];
                }
            }
        }
        Ok(c)
    }
    fn as_gpu_ibp_forward(&self) -> Option<&dyn ny_core::GpuIbpForward> {
        Some(self)
    }
}

#[test]
fn sequential_point_declines_whole_interval_gpu_and_uses_original_linear_gemm() {
    let network = cancellation_network();
    let input = BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[2, 512]), 1.0)).unwrap();
    for fail in [false, true] {
        let engine = ResidentPointEngine {
            fail,
            resident: AtomicUsize::new(0),
            gemms: AtomicUsize::new(0),
        };
        let point = propagate_concrete_point_plain(&network, &input, Some(&engine)).unwrap();
        assert!(point.lower().iter().all(|&v| v == 0.0));
        assert_eq!(point.lower(), point.upper());
        assert_eq!(engine.resident.load(Ordering::SeqCst), 0);
        assert_eq!(engine.gemms.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn sequential_point_causal_dispatch_computes_nonuniform_original_mask() {
    let mut network = Network::new();
    network.add_layer(Layer::CausalSoftmax(
        crate::layers::CausalSoftmaxLayer::new(-1),
    ));
    let input = BoundedTensor::concrete(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 1000.0, 0.0, 1.0]).unwrap(),
    )
    .unwrap();
    let point = propagate_concrete_point_plain(&network, &input, None).unwrap();
    assert_eq!(point.lower()[[0, 0]], 1.0);
    assert_eq!(point.lower()[[0, 1]], 0.0);
    assert_eq!(
        point.lower()[[1, 1]],
        (1.0 / (1.0 + (-1.0_f64).exp())) as f32
    );
    assert_eq!(point.lower(), point.upper());
}
