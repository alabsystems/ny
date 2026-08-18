// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared test helpers for CROWN test modules.

use crate::BoundedTensor;
use ny_core::{GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownResult, NaiveCpuGemmEngine};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
use std::time::Instant;

// Canonical CountingGemmEngine lives in ny-test-utils; re-export for
// existing callers that import from this module.
pub use ny_test_utils::CountingGemmEngine;

pub fn total_width(bounds: &BoundedTensor) -> f32 {
    bounds.width().iter().copied().sum()
}

/// Assert that a `BoundedTensor` contains only finite values (no NaN or Inf).
///
/// GPU CROWN engine-threading tests verify that bounds flowing through
/// the GemmEngine path remain finite. NaN/Inf in output bounds indicates
/// a numerical issue in the engine dispatch or backward propagation.
pub fn assert_bounds_finite(bounds: &BoundedTensor, label: &str) {
    assert!(
        !bounds.lower().iter().any(|v| v.is_nan() || v.is_infinite()),
        "{label}: lower bounds contain NaN/Inf"
    );
    assert!(
        !bounds.upper().iter().any(|v| v.is_nan() || v.is_infinite()),
        "{label}: upper bounds contain NaN/Inf"
    );
}

pub use ny_test_utils::assert_bounded_tensor_close;

/// Mock `GpuCrownBackward` engine for GPU CROWN fast-path tests.
///
/// Consolidates 3 near-identical implementations from `crown_ibp.rs`,
/// `gpu_fast_path.rs`, and `verifier/integration.rs`.
///
/// When `num_specs` matches the expected output dimension, returns configured
/// bounds. Otherwise returns conservative wide `[-1e6, +1e6]` bounds that let
/// IBP intersection dominate (soundness-preserving dummy result). See
/// `gpu_partial_oracle.rs` for a stricter queue-based alternative that panics
/// on unexpected calls.
pub struct MockGpuCrownEngine {
    expected_lower: Vec<f32>,
    expected_upper: Vec<f32>,
    fail_gpu: bool,
    gpu_calls: AtomicUsize,
    observed_num_specs: Mutex<Option<usize>>,
    observed_layer_kinds: Mutex<Option<Vec<&'static str>>>,
    crown_backward_deadline: Mutex<Option<Instant>>,
}

impl MockGpuCrownEngine {
    /// Create a mock engine that returns the given bounds on matching `num_specs`.
    pub fn succeed(expected: &BoundedTensor) -> Self {
        Self {
            expected_lower: expected.lower().iter().copied().collect(),
            expected_upper: expected.upper().iter().copied().collect(),
            fail_gpu: false,
            gpu_calls: AtomicUsize::new(0),
            observed_num_specs: Mutex::new(None),
            observed_layer_kinds: Mutex::new(None),
            crown_backward_deadline: Mutex::new(None),
        }
    }

    /// Alias for `succeed` — used by CROWN-IBP and verifier integration tests.
    pub fn from_expected(expected: &BoundedTensor) -> Self {
        Self::succeed(expected)
    }

    /// Create a mock engine that always returns an error.
    pub fn fail() -> Self {
        Self {
            expected_lower: Vec::new(),
            expected_upper: Vec::new(),
            fail_gpu: true,
            gpu_calls: AtomicUsize::new(0),
            observed_num_specs: Mutex::new(None),
            observed_layer_kinds: Mutex::new(None),
            crown_backward_deadline: Mutex::new(None),
        }
    }

    pub fn gpu_calls(&self) -> usize {
        self.gpu_calls.load(Ordering::SeqCst)
    }

    pub fn observed_num_specs(&self) -> Option<usize> {
        *self
            .observed_num_specs
            .lock()
            .expect("observed_num_specs mutex should not be poisoned")
    }

    pub fn observed_layer_kinds(&self) -> Option<Vec<&'static str>> {
        self.observed_layer_kinds
            .lock()
            .expect("observed_layer_kinds mutex should not be poisoned")
            .clone()
    }
}

impl GemmEngine for MockGpuCrownEngine {
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

impl GpuCrownBackward for MockGpuCrownEngine {
    fn crown_backward_gpu(
        &self,
        layers: &[GpuCrownLayer],
        _spec: &[f32],
        num_specs: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> ny_core::Result<GpuCrownResult> {
        if self
            .crown_backward_deadline
            .lock()
            .expect("crown_backward_deadline mutex should not be poisoned")
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(ny_core::NyError::DeadlineExceeded(
                "mock GPU CROWN deadline exceeded before launch".to_string(),
            ));
        }
        self.gpu_calls.fetch_add(1, Ordering::SeqCst);
        *self
            .observed_num_specs
            .lock()
            .expect("observed_num_specs mutex should not be poisoned") = Some(num_specs);
        *self
            .observed_layer_kinds
            .lock()
            .expect("observed_layer_kinds mutex should not be poisoned") =
            Some(gpu_crown_layer_kinds(layers));

        if self.fail_gpu {
            return Err(ny_core::NyError::UnsupportedConfiguration(
                "mock gpu failure".to_string(),
            ));
        }

        assert_eq!(
            input_lower.len(),
            input_upper.len(),
            "mock GPU engine expects matching input bounds lengths"
        );

        // Return configured bounds when num_specs matches expected output dim.
        // Otherwise return conservative wide bounds that let IBP intersection
        // dominate (#3599 Phase 1: varying num_specs for per-node CROWN-IBP).
        if num_specs == self.expected_lower.len() {
            Ok(GpuCrownResult {
                lower_bounds: self.expected_lower.clone(),
                upper_bounds: self.expected_upper.clone(),
            })
        } else {
            Ok(GpuCrownResult {
                lower_bounds: vec![-1e6; num_specs],
                upper_bounds: vec![1e6; num_specs],
            })
        }
    }

    fn set_crown_backward_deadline(&self, deadline: Option<Instant>) {
        *self
            .crown_backward_deadline
            .lock()
            .expect("crown_backward_deadline mutex should not be poisoned") = deadline;
    }

    fn honors_crown_backward_deadline(&self) -> bool {
        true
    }
}

/// Extract layer kind names from a slice of `GpuCrownLayer`.
pub fn gpu_crown_layer_kinds(layers: &[GpuCrownLayer]) -> Vec<&'static str> {
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
