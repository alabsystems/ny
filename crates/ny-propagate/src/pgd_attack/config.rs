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

/// Per-run policy for the exact point-VJP graph PGD routes.
///
/// VNN-COMP constructs this from its entry-time frozen [`ny_levers::LeverSet`]
/// and carries it through [`PgdConfig`]. Interactive/compatibility entry points
/// may instead call [`Self::legacy_env_over`] once at their configuration
/// boundary. The attack consumers themselves never reopen process-global
/// environment state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactVjpPgdConfig {
    /// Prefer exact-gradient conjunctive routing over generic batched SPSA.
    pub prefer_exact_batched: bool,
    /// Attempt the point-Jacobian fast gradient in sequential graph PGD.
    pub fast_gradient: bool,
    /// Admit both batched exact-VJP drivers.
    pub batch_enabled: bool,
    /// Requested wide restart width. `None` keeps the route-specific default.
    pub batch_width: Option<usize>,
    /// Uniform seed-replay lanes in the first conjunctive wave.
    pub legacy_lanes: usize,
    /// Output-diversified initialization steps per conjunctive wave.
    pub odi_steps: usize,
    /// ODI step scale in units of the input half-range.
    pub odi_eta: f32,
    /// Joint-margin steps per conjunctive wave. `None` inherits `num_steps`.
    pub wave_steps: Option<usize>,
    /// Rate-projection lane-recycling window; zero disables.
    pub lane_window: usize,
    /// Enable endgame exploit recycling.
    pub exploit: bool,
    /// Tau-hinge threshold for odd conjunctive lanes; zero disables.
    pub hinge_tau: f32,
}

impl Default for ExactVjpPgdConfig {
    fn default() -> Self {
        Self {
            prefer_exact_batched: true,
            fast_gradient: true,
            batch_enabled: true,
            batch_width: None,
            legacy_lanes: 0,
            odi_steps: 8,
            odi_eta: 1.0,
            wave_steps: None,
            lane_window: 0,
            exploit: true,
            hinge_tau: 0.0,
        }
    }
}

impl ExactVjpPgdConfig {
    /// Resolve this policy from the exact immutable set used for the run's
    /// lever receipt.
    pub fn from_levers(levers: &ny_levers::LeverSet) -> Result<Self, String> {
        Self::from_values(|decl| {
            levers
                .resolved(decl)
                .map(|resolved| resolved.value.clone())
                .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))
        })
    }

    /// Resolve the historical process environment over an already-typed
    /// interactive policy exactly once.
    ///
    /// This preserves all legacy parsing/default behavior, including
    /// route-specific contextual defaults and default-on gates for which only
    /// the exact token `0` kills. Scored VNN-COMP runs do not call this method;
    /// they use [`Self::from_levers`].
    pub fn legacy_env_over(self) -> Result<Self, String> {
        use ny_levers::decls::exact_vjp_pgd as d;

        self.validate()?;
        let registry = ny_levers::collect(&[&d::EXACT_VJP_PGD_LEVERS])
            .map_err(|error| format!("invalid exact-VJP lever registry: {error}"))?;
        let raw = ny_levers::RawLeverInputs::capture(&registry);
        let levers =
            ny_levers::LeverSet::resolve_layered(&registry, &raw, |decl| self.config_value(decl))
                .map_err(|error| format!("invalid typed exact-VJP compatibility policy: {error}"))?;
        Self::from_levers(&levers)
    }

    /// Validate direct programmatic construction at the same typed boundary as
    /// declaration/preset materialization.
    pub fn validate(self) -> Result<(), String> {
        if self.batch_width == Some(0) {
            return Err("exact-VJP batch width must be >= 1 when configured".to_string());
        }
        if self.wave_steps == Some(0) {
            return Err("exact-VJP wave steps must be >= 1 when configured".to_string());
        }
        if !self.odi_eta.is_finite() || self.odi_eta <= 0.0 {
            return Err(format!(
                "exact-VJP ODI eta must be finite and > 0, got {}",
                self.odi_eta
            ));
        }
        if !self.hinge_tau.is_finite() || self.hinge_tau < 0.0 {
            return Err(format!(
                "exact-VJP hinge tau must be finite and >= 0, got {}",
                self.hinge_tau
            ));
        }
        Ok(())
    }

    /// Effective conjunctive width before any adapter capacity retry.
    pub const fn conjunctive_batch_width(self) -> usize {
        match self.batch_width {
            Some(width) => width,
            None => 64,
        }
    }

    /// Effective disjunctive width.
    pub const fn disjunctive_batch_width(self) -> usize {
        match self.batch_width {
            Some(width) => width,
            None => 32,
        }
    }

    /// Effective conjunctive steps per wave.
    pub const fn effective_wave_steps(self, pgd_steps: usize) -> usize {
        match self.wave_steps {
            Some(steps) => steps,
            None => pgd_steps,
        }
    }

    fn config_value(self, decl: &'static ny_levers::LeverDecl) -> Option<ny_levers::LeverValue> {
        use ny_levers::decls::exact_vjp_pgd as d;
        use ny_levers::LeverValue;

        if std::ptr::eq(decl, &raw const d::EXACT_BATCHED) {
            Some(LeverValue::Bool(self.prefer_exact_batched))
        } else if std::ptr::eq(decl, &raw const d::FAST_GRAD) {
            Some(LeverValue::Bool(self.fast_gradient))
        } else if std::ptr::eq(decl, &raw const d::VJP_BATCH) {
            Some(LeverValue::Bool(self.batch_enabled))
        } else if std::ptr::eq(decl, &raw const d::VJP_K) {
            self.batch_width.map(|value| LeverValue::U64(value as u64))
        } else if std::ptr::eq(decl, &raw const d::LEGACY_LANES) {
            Some(LeverValue::U64(self.legacy_lanes as u64))
        } else if std::ptr::eq(decl, &raw const d::ODI_STEPS) {
            Some(LeverValue::U64(self.odi_steps as u64))
        } else if std::ptr::eq(decl, &raw const d::ODI_ETA) {
            Some(LeverValue::F64(f64::from(self.odi_eta)))
        } else if std::ptr::eq(decl, &raw const d::VJP_STEPS) {
            self.wave_steps.map(|value| LeverValue::U64(value as u64))
        } else if std::ptr::eq(decl, &raw const d::LANE_WINDOW) {
            Some(LeverValue::U64(self.lane_window as u64))
        } else if std::ptr::eq(decl, &raw const d::EXPLOIT) {
            Some(LeverValue::Bool(self.exploit))
        } else if std::ptr::eq(decl, &raw const d::HINGE_TAU) {
            Some(LeverValue::F64(f64::from(self.hinge_tau)))
        } else {
            None
        }
    }

    fn from_values<F>(mut value: F) -> Result<Self, String>
    where
        F: FnMut(&'static ny_levers::LeverDecl) -> Result<ny_levers::LeverValue, String>,
    {
        use ny_levers::decls::exact_vjp_pgd as d;
        use ny_levers::LeverValue;

        let boolean =
            |decl: &'static ny_levers::LeverDecl, value: LeverValue| -> Result<bool, String> {
                match value {
                    LeverValue::Bool(value) => Ok(value),
                    other => Err(format!(
                        "{} resolved to incompatible value {other:?}",
                        decl.name
                    )),
                }
            };
        let optional_usize = |decl: &'static ny_levers::LeverDecl,
                              value: LeverValue|
         -> Result<Option<usize>, String> {
            match value {
                LeverValue::Unset => Ok(None),
                LeverValue::U64(value) => usize::try_from(value).map(Some).map_err(|_| {
                    format!("{} resolved above this platform's usize range", decl.name)
                }),
                other => Err(format!(
                    "{} resolved to incompatible value {other:?}",
                    decl.name
                )),
            }
        };
        let required_usize =
            |decl: &'static ny_levers::LeverDecl, value: LeverValue| -> Result<usize, String> {
                optional_usize(decl, value)?.ok_or_else(|| {
                    format!(
                        "{} unexpectedly resolved without a concrete value",
                        decl.name
                    )
                })
            };
        let scalar_f32 =
            |decl: &'static ny_levers::LeverDecl, value: LeverValue| -> Result<f32, String> {
                match value {
                    LeverValue::F64(value) => {
                        let narrowed = value as f32;
                        if narrowed.is_finite() && f64::from(narrowed) == value {
                            Ok(narrowed)
                        } else {
                            Err(format!(
                                "{} did not resolve to an exact finite f32",
                                decl.name
                            ))
                        }
                    }
                    other => Err(format!(
                        "{} resolved to incompatible value {other:?}",
                        decl.name
                    )),
                }
            };

        let policy = Self {
            prefer_exact_batched: boolean(&d::EXACT_BATCHED, value(&d::EXACT_BATCHED)?)?,
            fast_gradient: boolean(&d::FAST_GRAD, value(&d::FAST_GRAD)?)?,
            batch_enabled: boolean(&d::VJP_BATCH, value(&d::VJP_BATCH)?)?,
            batch_width: optional_usize(&d::VJP_K, value(&d::VJP_K)?)?,
            legacy_lanes: required_usize(&d::LEGACY_LANES, value(&d::LEGACY_LANES)?)?,
            odi_steps: required_usize(&d::ODI_STEPS, value(&d::ODI_STEPS)?)?,
            odi_eta: scalar_f32(&d::ODI_ETA, value(&d::ODI_ETA)?)?,
            wave_steps: optional_usize(&d::VJP_STEPS, value(&d::VJP_STEPS)?)?,
            lane_window: required_usize(&d::LANE_WINDOW, value(&d::LANE_WINDOW)?)?,
            exploit: boolean(&d::EXPLOIT, value(&d::EXPLOIT)?)?,
            hinge_tau: scalar_f32(&d::HINGE_TAU, value(&d::HINGE_TAU)?)?,
        };
        policy.validate()?;
        Ok(policy)
    }
}

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
    /// Frozen per-run policy for batched exact point-VJP graph attacks.
    pub exact_vjp: ExactVjpPgdConfig,
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
            exact_vjp: ExactVjpPgdConfig::default(),
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

#[cfg(test)]
mod exact_vjp_policy_tests {
    use super::*;
    use ny_levers::decls::exact_vjp_pgd::EXACT_VJP_PGD_LEVERS;

    fn registry() -> ny_levers::LeverRegistry {
        ny_levers::collect(&[&EXACT_VJP_PGD_LEVERS]).unwrap()
    }

    #[test]
    fn defaults_keep_route_specific_contextual_width_and_steps() {
        let policy = ExactVjpPgdConfig::default();
        assert!(policy.fast_gradient);
        assert_eq!(policy.conjunctive_batch_width(), 64);
        assert_eq!(policy.disjunctive_batch_width(), 32);
        assert_eq!(policy.effective_wave_steps(1_000), 1_000);
        policy.validate().unwrap();
    }

    #[test]
    fn two_frozen_sets_produce_distinct_run_scoped_policies() {
        let first = ny_levers::LeverSet::freeze_with(&registry(), |name| match name {
            "NY_PGD_EXACT_BATCHED" => Some("0".to_string()),
            "NY_PGD_FAST_GRAD" => Some("0".to_string()),
            "NY_PGD_VJP_K" => Some("52".to_string()),
            "NY_PGD_ODI_ETA" => Some("0.5".to_string()),
            "NY_PGD_HINGE_TAU" => Some("0.0001".to_string()),
            _ => None,
        });
        let second = ny_levers::LeverSet::freeze_with(&registry(), |name| match name {
            "NY_PGD_EXACT_BATCHED" => Some("1".to_string()),
            "NY_PGD_FAST_GRAD" => Some("1".to_string()),
            "NY_PGD_VJP_K" => Some("96".to_string()),
            "NY_PGD_ODI_ETA" => Some("2".to_string()),
            _ => None,
        });

        let first = ExactVjpPgdConfig::from_levers(&first).unwrap();
        let second = ExactVjpPgdConfig::from_levers(&second).unwrap();
        assert!(!first.prefer_exact_batched);
        assert!(!first.fast_gradient);
        assert_eq!(first.batch_width, Some(52));
        assert_eq!(first.odi_eta, 0.5);
        assert_eq!(first.hinge_tau, 0.0001_f32);
        assert!(second.prefer_exact_batched);
        assert!(second.fast_gradient);
        assert_eq!(second.batch_width, Some(96));
        assert_eq!(second.odi_eta, 2.0);
        assert_eq!(second.hinge_tau, 0.0);
        assert_ne!(first, second, "per-run policy must not latch process-wide");
    }

    #[test]
    fn legacy_env_over_preserves_fast_gradient_parser_and_layering() {
        let _env_lock = ny_test_utils::env::lock_env();
        let registry = registry();
        let _unset: Vec<_> = registry
            .all()
            .iter()
            .map(|decl| ny_test_utils::env::ScopedEnvVar::unset(decl.name))
            .collect();
        let typed_off = ExactVjpPgdConfig {
            fast_gradient: false,
            ..ExactVjpPgdConfig::default()
        };

        assert!(!typed_off.legacy_env_over().unwrap().fast_gradient);
        {
            let _zero = ny_test_utils::env::ScopedEnvVar::set("NY_PGD_FAST_GRAD", "0");
            assert!(
                !ExactVjpPgdConfig::default()
                    .legacy_env_over()
                    .unwrap()
                    .fast_gradient
            );
        }
        {
            let _one = ny_test_utils::env::ScopedEnvVar::set("NY_PGD_FAST_GRAD", "1");
            assert!(typed_off.legacy_env_over().unwrap().fast_gradient);
        }
        {
            let _malformed = ny_test_utils::env::ScopedEnvVar::set("NY_PGD_FAST_GRAD", "false");
            assert!(
                typed_off.legacy_env_over().unwrap().fast_gradient,
                "a malformed present environment value suppresses typed false and restores the declaration's legacy-on default"
            );
        }
    }

    #[test]
    fn direct_typed_policy_rejects_values_the_declarations_reject() {
        for invalid in [
            ExactVjpPgdConfig {
                batch_width: Some(0),
                ..ExactVjpPgdConfig::default()
            },
            ExactVjpPgdConfig {
                wave_steps: Some(0),
                ..ExactVjpPgdConfig::default()
            },
            ExactVjpPgdConfig {
                odi_eta: f32::NAN,
                ..ExactVjpPgdConfig::default()
            },
            ExactVjpPgdConfig {
                odi_eta: 0.0,
                ..ExactVjpPgdConfig::default()
            },
            ExactVjpPgdConfig {
                hinge_tau: -1.0,
                ..ExactVjpPgdConfig::default()
            },
            ExactVjpPgdConfig {
                hinge_tau: f32::INFINITY,
                ..ExactVjpPgdConfig::default()
            },
        ] {
            assert!(invalid.validate().is_err(), "accepted {invalid:?}");
        }
    }
}
