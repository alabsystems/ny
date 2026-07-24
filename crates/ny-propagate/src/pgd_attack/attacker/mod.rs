// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PGD (Projected Gradient Descent) attacker: core utilities and single-output attack.
//!
//! Split from monolithic `attacker.rs` (1813 LOC) into a directory module
//! per design `designs/2026-03-19-issue-1948-pgd-attacker-directory-module-refresh.md`.

mod batched;
mod dense_sweep;
mod eval;
mod init;
mod points;
pub(super) mod restart;
mod spsa;
mod standard;

#[cfg(test)]
mod tests;

use ny_core::{GemmEngine, GpuIbpModelPlan, NyError, Result};
use ny_tensor::BoundedTensor;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::Mutex;

use super::config::PgdConfig;
use super::result::PgdResult;
use crate::Network;

pub(in crate::pgd_attack) use eval::output_value;

// Cache resident plans against the layer backing store rather than the outer
// Network allocation so reused attacker instances do not alias stale weights.
struct CachedGpuIbpPlanKey {
    layers_ptr: usize,
    layer_count: usize,
    input_shape: Vec<usize>,
}

impl CachedGpuIbpPlanKey {
    fn new(network: &Network, input_shape: &[usize]) -> Self {
        let layers = network.layers();
        Self {
            layers_ptr: layers.as_ptr() as usize,
            layer_count: layers.len(),
            input_shape: input_shape.to_vec(),
        }
    }
}

struct CachedGpuIbpPlanEntry {
    key: CachedGpuIbpPlanKey,
    plan: Option<Box<dyn GpuIbpModelPlan>>,
}

/// Engine adapter that redirects `gemm_f32` to the inner engine's
/// soundness-free `gemm_f32_fast` (tensor cores where available) and forwards
/// every other method unchanged. The attacker wraps its engine in this: all
/// attacker evals are attack-only (found candidates are re-checked
/// concretely), so the shared propagate plumbing can use the fast path without
/// bifurcating its call sites. MUST NOT be handed to verdict-feeding bound
/// computation — it deliberately breaks `gemm_f32`'s IEEE RN-f32 contract.
struct FastMathEngine<'a>(&'a dyn GemmEngine);

impl GemmEngine for FastMathEngine<'_> {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.0.gemm_f32_fast(m, k, n, a, b)
    }
    fn gemm_f32_fast(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>> {
        self.0.gemm_f32_fast(m, k, n, a, b)
    }
    fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        self.0.gemm_f64(m, k, n, a, b)
    }
    fn conv_transpose_2d(
        &self,
        a_reshaped: &[f32],
        weight_col: &[f32],
        params: &ny_core::ConvTranspose2dParams,
    ) -> Result<Vec<f32>> {
        self.0.conv_transpose_2d(a_reshaped, weight_col, params)
    }
    fn conv_transpose_2d_pair_cached(
        &self,
        a_lower: &[f32],
        a_upper: &[f32],
        weight_col: &std::sync::Arc<[f32]>,
        params: &ny_core::ConvTranspose2dParams,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.0
            .conv_transpose_2d_pair_cached(a_lower, a_upper, weight_col, params)
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn ny_core::GpuCrownBackward> {
        self.0.as_gpu_crown_backward()
    }
    fn as_gpu_ibp_forward(&self) -> Option<&dyn ny_core::GpuIbpForward> {
        self.0.as_gpu_ibp_forward()
    }
    fn as_gpu_ibp_forward_ext(&self) -> Option<&dyn ny_core::GpuIbpForwardExt> {
        self.0.as_gpu_ibp_forward_ext()
    }
    fn as_gpu_dag_ibp_forward_ext(&self) -> Option<&dyn ny_core::GpuDagIbpForwardExt> {
        self.0.as_gpu_dag_ibp_forward_ext()
    }
    fn gemm_interval_sound(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a_lo: &[f32],
        a_hi: &[f32],
        b_lo: &[f32],
        b_hi: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        // Forward to the inner engine's EXACT implementation: interval results
        // are labeled sound even inside attack heuristics, so keep them so.
        self.0.gemm_interval_sound(m, k, n, a_lo, a_hi, b_lo, b_hi)
    }
    fn crown_aw_error_step(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        a_err: &[f32],
        w: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.0.crown_aw_error_step(m, k, n, a, a_err, w)
    }
}

/// PGD attacker for finding counterexamples.
pub struct PgdAttacker<'a> {
    config: PgdConfig,
    engine: Option<&'a dyn GemmEngine>,
    /// `engine` wrapped in [`FastMathEngine`]; what the eval paths actually use.
    fast_engine: Option<FastMathEngine<'a>>,
    model_plan: Mutex<Option<CachedGpuIbpPlanEntry>>,
}

impl PgdAttacker<'static> {
    /// Create a new PGD attacker with the given configuration.
    pub fn new(config: PgdConfig) -> Self {
        Self {
            config,
            engine: None,
            fast_engine: None,
            model_plan: Mutex::new(None),
        }
    }
}

impl<'a> PgdAttacker<'a> {
    /// Create a new PGD attacker with an optional borrowed GEMM engine.
    pub fn new_with_optional_engine(config: PgdConfig, engine: Option<&'a dyn GemmEngine>) -> Self {
        Self {
            config,
            engine,
            fast_engine: engine.map(FastMathEngine),
            model_plan: Mutex::new(None),
        }
    }

    /// Attach a GEMM engine for engine-aware IBP evaluation.
    pub fn with_engine(mut self, engine: &'a dyn GemmEngine) -> Self {
        self.engine = Some(engine);
        self.fast_engine = Some(FastMathEngine(engine));
        self.model_plan = Mutex::new(None);
        self
    }

    /// The engine attack evals run on: the attached engine behind the
    /// fast-math redirect (tensor-core `gemm_f32_fast`), since every attacker
    /// eval is soundness-free (candidates are re-checked concretely).
    pub(super) fn eval_engine(&self) -> Option<&dyn GemmEngine> {
        self.fast_engine.as_ref().map(|f| f as &dyn GemmEngine)
    }

    /// Access the configuration.
    pub(super) fn config(&self) -> &PgdConfig {
        &self.config
    }

    /// Create a seeded RNG for a restart.
    pub(super) fn seeded_rng(&self, seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    /// Run PGD attack to find counterexample where `output[output_idx]` violates threshold.
    ///
    /// For `verify_upper_bound = true`: looking for output >= threshold (property violation)
    /// For `verify_upper_bound = false`: looking for output <= threshold (property violation)
    pub fn attack(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx: usize,
        threshold: f32,
        verify_upper_bound: bool,
    ) -> Result<PgdResult> {
        // Dense low-effective-dimension sweep pre-phase (#dense-sweep,
        // config-gated): when at most `dense_sweep_max_dims` input dims have
        // nonzero width, a deterministic grid + top-k refinement covers the
        // whole box better than gradient restarts. A hit short-circuits PGD;
        // otherwise the normal restart schedule below runs unchanged.
        if let Some((input, output, evals)) = self.try_dense_low_dim_sweep(
            network,
            input_bounds,
            output_idx,
            threshold,
            verify_upper_bound,
        )? {
            let value = output_value(&output, output_idx)?;
            return Ok(PgdResult {
                found_counterexample: true,
                counterexample: Some(input),
                output: Some(output),
                best_output_value: value,
                restarts_completed: 0,
                failed_restarts: 0,
                total_evaluations: evals,
            });
        }
        // Batched whenever there are multiple restarts, engine or not
        // (reference: alpha-beta-CROWN attack_pgd.py:267, batched on CPU too).
        // With an engine this folds N*S independent dispatches into S+1 batched
        // dispatches (and large batched GEMMs reach the cuBLAS f32 seam); the
        // engine-less per-layer fallback still folds all restarts into ONE
        // vectorized CPU matmul per layer, beating per-restart scalar forwards.
        // NY_PGD_NO_CPU_BATCH restores the old engine-gated selection (A/B
        // escape hatch while the CPU-batched path is being validated broadly).
        let batch_allowed =
            self.engine.is_some() || std::env::var_os("NY_PGD_NO_CPU_BATCH").is_none();
        if batch_allowed && self.config.num_restarts > 1 {
            self.attack_batched(
                network,
                input_bounds,
                output_idx,
                threshold,
                verify_upper_bound,
            )
            .map_err(|e| {
                // Restarts advance in lockstep, so a terminal batched error IS
                // an all-restarts failure; surface the same #3096 contract the
                // sequential/parallel paths use (callers and tests match on it).
                NyError::InternalError(format!(
                    "PGD attack: all {} restarts failed (batched lockstep). Last error: {e}",
                    self.config.num_restarts
                ))
            })
        } else if self.config.parallel && self.config.num_restarts >= 10 {
            self.attack_parallel(
                network,
                input_bounds,
                output_idx,
                threshold,
                verify_upper_bound,
            )
        } else {
            self.attack_sequential(
                network,
                input_bounds,
                output_idx,
                threshold,
                verify_upper_bound,
            )
        }
    }
}
