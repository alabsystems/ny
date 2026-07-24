// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared test GEMM engine that counts calls while delegating to the CPU backend.
//!
//! Consolidates ~25 near-identical `CountingGemmEngine` definitions scattered
//! across test modules in ny-propagate, ny-cli, ny-onnx, and ny-api.
//! See issue #4141 for the deduplication tracker.

use ny_core::{GemmEngine, NaiveCpuGemmEngine};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

/// A [`GemmEngine`] that delegates to [`NaiveCpuGemmEngine`] while counting calls.
///
/// Used by engine-threading tests to verify that the GEMM engine parameter is
/// actually threaded through propagation paths. Wraps the call counter in
/// [`Arc`] so the engine can be cloned across threads (e.g., parallel BaB tests).
#[derive(Clone, Default)]
pub struct CountingGemmEngine {
    gemm_calls: Arc<AtomicUsize>,
}

impl CountingGemmEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of `gemm_f32` calls observed so far.
    pub fn gemm_calls(&self) -> usize {
        self.gemm_calls.load(Ordering::SeqCst)
    }
}

impl GemmEngine for CountingGemmEngine {
    fn gemm_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        self.gemm_calls.fetch_add(1, Ordering::SeqCst);
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counting_gemm_engine_counts_calls() {
        let engine = CountingGemmEngine::new();
        assert_eq!(engine.gemm_calls(), 0);

        // 2x2 identity multiplication
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let result = engine.gemm_f32(2, 2, 2, &a, &b);
        assert!(result.is_ok());
        assert_eq!(engine.gemm_calls(), 1);

        let _ = engine.gemm_f32(2, 2, 2, &a, &b);
        assert_eq!(engine.gemm_calls(), 2);
    }

    #[test]
    fn test_counting_gemm_engine_clone_shares_counter() {
        let engine = CountingGemmEngine::new();
        let clone = engine.clone();

        let a = vec![1.0];
        let b = vec![1.0];
        let _ = engine.gemm_f32(1, 1, 1, &a, &b);
        assert_eq!(clone.gemm_calls(), 1);
    }
}
