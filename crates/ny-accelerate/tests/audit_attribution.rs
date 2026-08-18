// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Deterministic routing contract behind the opt-in attribution measurement.

use std::sync::atomic::{AtomicBool, Ordering};

use ny_core::{GemmEngine, NyError, Result};

struct AttributionEngine {
    enabled: AtomicBool,
}

impl AttributionEngine {
    fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
}

impl GemmEngine for AttributionEngine {
    fn backend_provenance(&self) -> &'static str {
        "deterministic-attribution-test"
    }

    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> Result<Vec<f32>> {
        Err(NyError::UnsupportedOp(
            "f32 is outside this contract".into(),
        ))
    }

    fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        if !self.enabled.load(Ordering::Relaxed) {
            return Err(NyError::UnsupportedOp("attribution arm is off".into()));
        }
        if a.len() != m * k || b.len() != k * n {
            return Err(NyError::InvalidSpec(
                "attribution test shape mismatch".into(),
            ));
        }
        let mut out = vec![0.0; m * n];
        for row in 0..m {
            for col in 0..n {
                out[row * n + col] = (0..k)
                    .map(|inner| a[row * k + inner] * b[inner * n + col])
                    .sum();
            }
        }
        Ok(out)
    }
}

#[test]
fn attribution_arms_are_observably_distinct_and_arithmetically_exact() {
    let engine = AttributionEngine {
        enabled: AtomicBool::new(false),
    };
    let lhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    assert!(matches!(
        engine.gemm_f64(2, 3, 2, &lhs, &rhs),
        Err(NyError::UnsupportedOp(_))
    ));
    engine.set_enabled(true);
    assert_eq!(
        engine.gemm_f64(2, 3, 2, &lhs, &rhs).expect("enabled arm"),
        vec![58.0, 64.0, 139.0, 154.0]
    );
}
