// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ndarray::{array, ArrayD, IxDyn};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;

use super::super::{LinearLayer, LinearPointArm};

struct SequentialF32(AtomicUsize);
impl GemmEngine for SequentialF32 {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        let mut values = vec![0.0; m * n];
        for row in 0..m {
            for out in 0..n {
                for column in 0..k {
                    values[row * n + out] += a[row * k + column] * b[column * n + out];
                }
            }
        }
        Ok(values)
    }
}

#[test]
fn linear_point_binade_values_are_not_interval_centers() {
    let layer = LinearLayer::from_flat(vec![1.0 / 512.0; 512], 1, 512, None).unwrap();
    let input = BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[2, 512]), 1.0)).unwrap();
    let engine = SequentialF32(AtomicUsize::new(0));
    for backend in [None, Some(&engine as &dyn GemmEngine)] {
        let point = layer
            .propagate_concrete_point_with_engine_and_deadline(&input, backend, None)
            .unwrap();
        assert_eq!(point.shape(), &[2, 1]);
        assert!(point.lower().iter().all(|&v| v == 1.0));
        assert_eq!(point.lower(), point.upper());
    }
    assert_eq!(engine.0.load(Ordering::SeqCst), 1);
    use crate::layers::common::BoundPropagation;
    let interval = layer.propagate_ibp(&input).unwrap();
    assert!(interval.lower().iter().all(|&v| v < 1.0));
    assert!(interval.upper().iter().all(|&v| v > 1.0));
    assert!(interval.center().iter().any(|&v| v != 1.0));
}

#[test]
fn linear_point_cancellation_never_substitutes_for_an_interval_certificate() {
    let layer = LinearLayer::new(array![[16_777_216.0, 1.0, -16_777_216.0]], None).unwrap();
    let input = BoundedTensor::concrete(array![[1.0, 1.0, 1.0]].into_dyn()).unwrap();
    let engine = SequentialF32(AtomicUsize::new(0));
    let observation = layer
        .propagate_concrete_point_with_engine_and_deadline(&input, Some(&engine), None)
        .unwrap();
    assert_eq!(observation.lower()[[0, 0]], 0.0);
    assert_eq!(engine.0.load(Ordering::SeqCst), 1);
    let enclosure = layer
        .propagate_ibp_with_engine(&input, Some(&engine))
        .unwrap();
    assert!(enclosure.lower()[[0, 0]] <= 1.0 && enclosure.upper()[[0, 0]] >= 1.0);
    assert!(enclosure.lower()[[0, 0]] < enclosure.upper()[[0, 0]]);
    let direct = layer
        .propagate_concrete_point_with_engine_and_deadline(
            &input,
            Some(&engine),
            Some(Instant::now() + Duration::from_secs(30)),
        )
        .unwrap();
    assert_eq!(direct.lower()[[0, 0]], 1.0);
    assert_eq!(
        engine.0.load(Ordering::SeqCst),
        2,
        "finite route must not call an opaque GEMM"
    );
}

#[test]
fn linear_point_admission_rejects_boxes_shapes_nonfinite_and_expired_deadlines() {
    let layer = LinearLayer::new(array![[2.0, -1.0]], Some(array![0.25])).unwrap();
    let wrong = BoundedTensor::concrete(array![1.0].into_dyn()).unwrap();
    assert!(matches!(
        layer.propagate_concrete_point_with_engine_and_deadline(&wrong, None, None),
        Err(NyError::ShapeMismatch { .. })
    ));
    let bounds =
        BoundedTensor::new(array![1.0, 2.0].into_dyn(), array![1.0, 3.0].into_dyn()).unwrap();
    assert!(layer
        .propagate_concrete_point_with_engine_and_deadline(&bounds, None, None)
        .is_err());
    let nonfinite = BoundedTensor::new_allow_infinite(
        array![1.0, f32::INFINITY].into_dyn(),
        array![1.0, f32::INFINITY].into_dyn(),
    )
    .unwrap();
    assert!(layer
        .propagate_concrete_point_with_engine_and_deadline(&nonfinite, None, None)
        .is_err());
    let input = BoundedTensor::concrete(array![[1.0, 2.0]].into_dyn()).unwrap();
    let engine = SequentialF32(AtomicUsize::new(0));
    assert!(matches!(
        layer.propagate_concrete_point_with_engine_and_deadline(
            &input,
            Some(&engine),
            Some(Instant::now())
        ),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert_eq!(engine.0.load(Ordering::SeqCst), 0);
}

#[test]
fn linear_point_strided_inputs_preserve_logical_shape_and_bias() {
    let layer = LinearLayer::new(array![[2.0, -1.0]], Some(array![0.25])).unwrap();
    let input =
        BoundedTensor::concrete(array![[1.0, 3.0], [2.0, 4.0]].reversed_axes().into_dyn()).unwrap();
    let point = layer
        .propagate_concrete_point_with_engine_and_deadline(&input, None, None)
        .unwrap();
    assert_eq!(point.shape(), &[2, 1]);
    assert_eq!(point.lower(), &array![[0.25], [2.25]].into_dyn());
}

struct DecliningEngine;
impl GemmEngine for DecliningEngine {
    fn backend_provenance(&self) -> &'static str {
        "deliberately-declining-test-engine"
    }
    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        Err(NyError::UnsupportedOp("injected engine refusal".into()))
    }
}

#[test]
fn linear_point_receipt_records_the_completed_arm_and_actual_backend_fallback() {
    let layer = LinearLayer::from_flat(vec![1.0 / 512.0; 512], 1, 512, None).unwrap();
    let input = BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[1, 512]), 1.0)).unwrap();
    let engine = SequentialF32(AtomicUsize::new(0));
    let (_, cpu) = layer
        .propagate_concrete_point_with_receipt(&input, None, None)
        .unwrap();
    let (_, external) = layer
        .propagate_concrete_point_with_receipt(&input, Some(&engine), None)
        .unwrap();
    let (_, fallback) = layer
        .propagate_concrete_point_with_receipt(&input, Some(&DecliningEngine), None)
        .unwrap();
    let (_, direct) = layer
        .propagate_concrete_point_with_receipt(
            &input,
            Some(&engine),
            Some(Instant::now() + Duration::from_secs(30)),
        )
        .unwrap();
    assert_eq!(cpu.arm, LinearPointArm::CpuFaerF32);
    assert_eq!(external.arm, LinearPointArm::EngineF32);
    assert_eq!(external.backend, "unspecified-custom-gemm-engine");
    assert_eq!(fallback.arm, LinearPointArm::CpuFaerF32AfterEngineRefusal);
    assert_eq!(fallback.backend, cpu.backend);
    assert_eq!(
        fallback.declined_backend,
        Some("deliberately-declining-test-engine")
    );
    assert_eq!(direct.arm, LinearPointArm::ScalarF64);
    assert_eq!(direct.declined_backend, None);
    assert_eq!(
        engine.0.load(Ordering::SeqCst),
        1,
        "finite arm cannot claim an unused engine"
    );
}

#[test]
fn linear_point_receipt_distinguishes_computations_even_when_bits_agree() {
    let layer = LinearLayer::new(array![[2.0, -1.0]], Some(array![0.25])).unwrap();
    let input = BoundedTensor::concrete(array![[1.0, 2.0]].into_dyn()).unwrap();
    let (binary32, a) = layer
        .propagate_concrete_point_with_receipt(&input, None, None)
        .unwrap();
    let (binary64, b) = layer
        .propagate_concrete_point_with_receipt(
            &input,
            None,
            Some(Instant::now() + Duration::from_secs(30)),
        )
        .unwrap();
    assert_eq!(binary32.lower(), binary64.lower());
    assert_ne!(a.arm.identity_name(), b.arm.identity_name());
}
