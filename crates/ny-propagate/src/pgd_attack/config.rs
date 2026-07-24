// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Configuration for PGD attacks.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use ny_tensor::BoundedTensor;

use super::optimizer::{auto_alpha, AdamClippingParams, PgdAlphaMode, PgdOptimizer, PgdStepState};

/// Default GAMA guidance weight λ₀ (Sriramanan et al., NeurIPS 2020, §4:
/// λ = 50 for softmax-space guidance; annealed linearly to 0).
pub const GAMA_LAMBDA_DEFAULT: f32 = 50.0;

/// Initialization strategy for PGD restarts.
///
/// Reference: alpha-beta-CROWN `attack_interface.py:29-35`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PgdInitialization {
    #[default]
    Uniform,
    Osi,
}

/// Configuration for PGD attack.
#[derive(Debug, Clone)]
pub struct PgdConfig {
    pub num_restarts: usize,
    pub num_steps: usize,
    /// Legacy scalar fallback retained for compatibility with older tests and
    /// call sites. New code should prefer `alpha_mode`.
    pub step_size: f32,
    pub spsa_delta: f32,
    pub seed: u64,
    pub parallel: bool,
    pub deadline: Option<Instant>,
    pub restart_when_stuck: bool,
    pub initialization: PgdInitialization,
    pub osi_steps: usize,
    pub optimizer: PgdOptimizer,
    pub alpha_mode: PgdAlphaMode,
    pub adam: AdamClippingParams,
    /// GAMA guidance weight λ₀ (#1449, `attack_mode: diversed_GAMA_PGD`).
    ///
    /// `Some(λ₀)` makes relational-target attack steps ascend the GAMA loss
    /// `softmax_margin + λ·‖P − softmax‖²` (Sriramanan et al., NeurIPS 2020)
    /// with λ annealed linearly from λ₀ to 0. Conjunctive graph attacks,
    /// including the generic restart-batched SPSA route and constant-threshold
    /// objectives, retain their raw property margin and add only the same
    /// annealed guidance term.
    /// `None` (default) keeps the raw-margin objective. Attack-only: it can
    /// only produce counterexample candidates, never affect a sound verdict.
    /// Reference: alpha-beta-CROWN `attack_mode: diversed_GAMA_PGD` →
    /// `GAMA_loss=True` (`attack_interface.py:29-35`).
    pub gama_lambda: Option<f32>,
    /// Straight-through-estimator surrogate gradient for `Layer::Sign`
    /// (#surrogate-sign, `attack: surrogate_sign_gradient`).
    ///
    /// `true` makes ATTACK gradient estimation treat `d/dx sign(x)` as 1
    /// (plain STE: probes are evaluated through a network whose Sign layers
    /// are replaced by the identity). The default `false` keeps the legacy
    /// `tanh(β·x)` smooth relaxation, which saturates to a zero gradient once
    /// BNN pre-activations leave `[-1, 1]` scale (traffic_signs QConv nets).
    /// Violation checks always use the TRUE Sign forward, and every candidate
    /// is re-validated downstream, so this can never affect a sound verdict.
    pub surrogate_sign_gradient: bool,
    /// Dense deterministic grid sweep for low-effective-dimension boxes
    /// (#dense-sweep, `attack: dense_low_dim_sweep`).
    ///
    /// `true` runs a pre-PGD phase when at most `dense_sweep_max_dims` input
    /// dims have nonzero width (cctsdb_yolo has 2): a uniform grid over the
    /// varying dims plus top-k local refinement, batched through the existing
    /// concrete forward machinery. Attack-only: any hit is a candidate for
    /// the normal witness path.
    pub dense_low_dim_sweep: bool,
    /// Maximum number of nonzero-width input dims for the dense sweep to run.
    pub dense_sweep_max_dims: usize,
    /// Total forward-evaluation budget for the dense sweep (grid + refinement).
    pub dense_sweep_points: usize,
}

impl Default for PgdConfig {
    fn default() -> Self {
        Self {
            num_restarts: 100,
            num_steps: 50,
            step_size: 0.01,
            spsa_delta: 0.001,
            seed: 42,
            parallel: true,
            deadline: None,
            restart_when_stuck: false,
            initialization: PgdInitialization::Uniform,
            osi_steps: 20,
            optimizer: PgdOptimizer::AdamClipping,
            alpha_mode: PgdAlphaMode::Auto,
            adam: AdamClippingParams::default(),
            gama_lambda: None,
            surrogate_sign_gradient: false,
            dense_low_dim_sweep: false,
            dense_sweep_max_dims: 3,
            dense_sweep_points: 32_768,
        }
    }
}

impl PgdConfig {
    pub fn past_deadline(&self) -> bool {
        self.deadline.map(|d| Instant::now() >= d).unwrap_or(false)
    }

    pub fn fast() -> Self {
        Self {
            num_restarts: 10,
            num_steps: 20,
            step_size: 0.01,
            spsa_delta: 0.001,
            seed: 42,
            parallel: false,
            deadline: None,
            restart_when_stuck: false,
            initialization: PgdInitialization::Uniform,
            osi_steps: 20,
            optimizer: PgdOptimizer::SignedGradient,
            alpha_mode: PgdAlphaMode::Scalar(0.01),
            adam: AdamClippingParams::default(),
            gama_lambda: None,
            ..Default::default()
        }
    }

    pub fn thorough() -> Self {
        Self {
            num_restarts: 1000,
            num_steps: 100,
            step_size: 0.005,
            spsa_delta: 0.0005,
            seed: 42,
            parallel: true,
            deadline: None,
            restart_when_stuck: false,
            initialization: PgdInitialization::Uniform,
            osi_steps: 20,
            optimizer: PgdOptimizer::AdamClipping,
            alpha_mode: PgdAlphaMode::Auto,
            adam: AdamClippingParams::default(),
            gama_lambda: None,
            ..Default::default()
        }
    }

    pub fn acas_xu() -> Self {
        Self {
            num_restarts: 5000,
            num_steps: 50,
            step_size: 0.01,
            spsa_delta: 0.001,
            seed: 42,
            parallel: true,
            deadline: None,
            restart_when_stuck: true,
            initialization: PgdInitialization::Uniform,
            osi_steps: 20,
            optimizer: PgdOptimizer::AdamClipping,
            alpha_mode: PgdAlphaMode::Auto,
            adam: AdamClippingParams::default(),
            gama_lambda: None,
            ..Default::default()
        }
    }

    pub(crate) fn create_step_state(&self, input_bounds: &BoundedTensor) -> PgdStepState {
        PgdStepState::from_config(
            self.optimizer,
            self.alpha_mode,
            self.step_size,
            self.adam,
            input_bounds,
            input_bounds.shape(),
        )
    }

    pub fn base_alpha(&self, input_bounds: &BoundedTensor) -> f32 {
        match self.alpha_mode {
            PgdAlphaMode::Auto => auto_alpha(input_bounds),
            PgdAlphaMode::Scalar(alpha) => alpha,
            PgdAlphaMode::InputRangeScaled(alpha) => {
                input_bounds
                    .upper()
                    .iter()
                    .zip(input_bounds.lower().iter())
                    .map(|(u, l)| (u - l).abs())
                    .fold(0.0_f32, f32::max)
                    * alpha
            }
        }
    }

    pub fn suggested_spsa_delta(&self, input_bounds: &BoundedTensor) -> f32 {
        (self.base_alpha(input_bounds) * 0.1).max(0.001)
    }
}
