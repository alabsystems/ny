// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{PyBranchingHeuristic, PyKfsbReduceOp};
use ny_propagate::{BetaCrownConfig as RustBetaCrownConfig, BranchingHeuristic, KfsbReduceOp};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Configuration for β-CROWN branch-and-bound verification.
///
/// Exposes advanced β-CROWN parameters for fine-grained control over
/// the verification process.
///
/// Example:
///     >>> config = ny.BetaCrownConfig()
///     >>> config.branching = ny.BranchingHeuristic.Kfsb
///     >>> config.fsb_candidates = 8
///     >>> config.enable_proactive_cuts = True
///     >>> result = ny.verify("model.onnx", method="beta", beta_config=config)
#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub struct BetaCrownConfig {
    /// Maximum number of domains to explore.
    #[pyo3(get, set)]
    pub max_domains: usize,
    /// Timeout in seconds.
    #[pyo3(get, set)]
    pub timeout_secs: u64,
    /// Maximum search tree depth.
    #[pyo3(get, set)]
    pub max_depth: usize,
    /// Use α-CROWN optimization within each domain.
    #[pyo3(get, set)]
    pub use_alpha_crown: bool,
    /// Reuse forward-linear intermediate bounds in the final backward pass.
    #[pyo3(get, set)]
    pub use_forward_bounds: bool,
    /// Use CROWN-IBP for tighter intermediate bounds.
    #[pyo3(get, set)]
    pub use_crown_ibp: bool,
    /// Branching heuristic.
    #[pyo3(get, set)]
    pub branching: PyBranchingHeuristic,
    /// Number of candidate neurons for FSB/kFSB.
    #[pyo3(get, set)]
    pub fsb_candidates: usize,
    /// Reduce operation for kFSB branching.
    #[pyo3(get, set)]
    pub kfsb_reduce_op: PyKfsbReduceOp,
    /// Learning rate for β optimization.
    #[pyo3(get, set)]
    pub beta_lr: f32,
    /// Number of β optimization iterations.
    #[pyo3(get, set)]
    pub beta_iterations: usize,
    /// β optimization tolerance.
    #[pyo3(get, set)]
    pub beta_tolerance: f32,
    /// Learning rate for α optimization.
    #[pyo3(get, set)]
    pub alpha_lr: f32,
    /// Number of domains to process in parallel.
    #[pyo3(get, set)]
    pub batch_size: usize,
    /// Enable GCP-CROWN cutting planes.
    #[pyo3(get, set)]
    pub enable_cuts: bool,
    /// Maximum number of cutting planes to retain.
    #[pyo3(get, set)]
    pub max_cuts: usize,
    /// Enable proactive cut generation (BICCOS-lite).
    #[pyo3(get, set)]
    pub enable_proactive_cuts: bool,
    /// Maximum number of proactive cuts.
    #[pyo3(get, set)]
    pub max_proactive_cuts: usize,
    /// Enable BICCOS constraint strengthening for verified-domain cuts.
    #[pyo3(get, set)]
    pub enable_biccos_constraint_strengthening: bool,
    /// Drop ratio for BICCOS constraint strengthening (quantile over influence scores).
    #[pyo3(get, set)]
    pub biccos_drop_ratio: f32,
    /// Enable BICCOS cold-start gating for cut generation.
    #[pyo3(get, set)]
    pub enable_biccos_cold_start: bool,
    /// Minimum verified domains before enabling cuts.
    #[pyo3(get, set)]
    pub biccos_min_verified: usize,
    /// Minimum verified domain rate before enabling cuts.
    #[pyo3(get, set)]
    pub biccos_min_verified_rate: f32,
    /// Sliding window size for verified-rate computation.
    #[pyo3(get, set)]
    pub biccos_verified_rate_window: usize,
    /// Minimum cuts generated before enabling cuts.
    #[pyo3(get, set)]
    pub biccos_min_cuts: usize,
    /// Minimum average bound gain per split before enabling cuts.
    #[pyo3(get, set)]
    pub biccos_min_bound_gain: f32,
    /// Sliding window size for bound-gain computation.
    #[pyo3(get, set)]
    pub biccos_bound_gain_window: usize,
    /// Maximum number of cold-start iterations before declaring exhaustion.
    #[pyo3(get, set)]
    pub biccos_cold_max_iters: usize,
    /// Maximum number of iterations to keep cut generation enabled.
    #[pyo3(get, set)]
    pub biccos_cut_window: usize,
    /// Minimum cut yield before disabling new cut generation.
    #[pyo3(get, set)]
    pub biccos_min_cut_yield: f32,
    /// Sliding window size for cut-yield computation.
    #[pyo3(get, set)]
    pub biccos_cut_yield_window: usize,
    /// Number of low-yield windows before disabling cut generation.
    #[pyo3(get, set)]
    pub biccos_cut_yield_patience: usize,
    /// Verify upper bound instead of lower bound (output < threshold).
    #[pyo3(get, set)]
    pub verify_upper_bound: bool,
    /// Enable PGD attack for counterexample finding.
    #[pyo3(get, set)]
    pub enable_pgd_attack: bool,
    /// Number of PGD restarts.
    #[pyo3(get, set)]
    pub pgd_restarts: usize,
    /// Number of PGD steps per restart.
    #[pyo3(get, set)]
    pub pgd_steps: usize,
    /// Enable relaxed clipping to tighten input bounds using CROWN constraints.
    /// Requires `branching` to be `InputSplit` to have effect.
    #[pyo3(get, set)]
    pub enable_relaxed_clip: bool,
    /// Number of relaxed clipping iterations per input split.
    #[pyo3(get, set)]
    pub relaxed_clip_iterations: usize,
    /// Enable static intermediate bound transfer in batched domains.
    #[pyo3(get, set)]
    pub enable_interm_transfer: bool,
    /// Enable intermediate domain clipping (clip_interm_domain).
    /// Requires ReLU branching (not input split) with split history constraints.
    #[pyo3(get, set)]
    pub enable_clip_interm_domain: bool,
    /// Number of objective neurons per layer to tighten (clip_interm_domain).
    #[pyo3(get, set)]
    pub clip_interm_topk: usize,
    /// Apply clip_interm_domain during alpha-CROWN optimization.
    #[pyo3(get, set)]
    pub clip_in_alpha_crown: bool,
    /// Prune infeasible domains during activation-space clipping.
    #[pyo3(get, set)]
    pub clip_interm_prune: bool,
    /// Use final-layer constraints when pruning clipped domains.
    #[pyo3(get, set)]
    pub clip_interm_use_final_layer: bool,
}

#[pymethods]
impl BetaCrownConfig {
    /// Create a new BetaCrownConfig with default values.
    #[new]
    pub(crate) fn new() -> Self {
        let rust_default = RustBetaCrownConfig::default();
        Self::from_rust(&rust_default)
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "BetaCrownConfig(max_domains={}, timeout_secs={}s, max_depth={}, use_alpha_crown={}, \
             use_forward_bounds={}, use_crown_ibp={}, branching={}, fsb_candidates={}, kfsb_reduce_op={}, beta_lr={:.3e}, \
             beta_iterations={}, beta_tolerance={:.3e}, alpha_lr={:.3e}, batch_size={}, \
             enable_cuts={}, max_cuts={}, enable_proactive_cuts={}, max_proactive_cuts={}, \
             biccos_strengthen={}, biccos_drop_ratio={}, biccos_cold_start={}, \
             biccos_min_verified={}, biccos_min_verified_rate={:.3}, biccos_verified_rate_window={}, \
             biccos_min_cuts={}, biccos_min_bound_gain={:.3e}, biccos_bound_gain_window={}, \
             biccos_cold_max_iters={}, biccos_cut_window={}, biccos_min_cut_yield={:.3}, \
             biccos_cut_yield_window={}, biccos_cut_yield_patience={}, verify_upper_bound={}, \
             enable_pgd_attack={}, pgd_restarts={}, pgd_steps={}, enable_relaxed_clip={}, \
             relaxed_clip_iterations={}, enable_clip_interm_domain={}, clip_interm_topk={}, \
             clip_in_alpha_crown={}, clip_interm_prune={}, clip_interm_use_final_layer={}, \
             enable_interm_transfer={})",
            self.max_domains,
            self.timeout_secs,
            self.max_depth,
            self.use_alpha_crown,
            self.use_forward_bounds,
            self.use_crown_ibp,
            self.branching.__repr__(),
            self.fsb_candidates,
            self.kfsb_reduce_op.__repr__(),
            self.beta_lr,
            self.beta_iterations,
            self.beta_tolerance,
            self.alpha_lr,
            self.batch_size,
            self.enable_cuts,
            self.max_cuts,
            self.enable_proactive_cuts,
            self.max_proactive_cuts,
            self.enable_biccos_constraint_strengthening,
            self.biccos_drop_ratio,
            self.enable_biccos_cold_start,
            self.biccos_min_verified,
            self.biccos_min_verified_rate,
            self.biccos_verified_rate_window,
            self.biccos_min_cuts,
            self.biccos_min_bound_gain,
            self.biccos_bound_gain_window,
            self.biccos_cold_max_iters,
            self.biccos_cut_window,
            self.biccos_min_cut_yield,
            self.biccos_cut_yield_window,
            self.biccos_cut_yield_patience,
            self.verify_upper_bound,
            self.enable_pgd_attack,
            self.pgd_restarts,
            self.pgd_steps,
            self.enable_relaxed_clip,
            self.relaxed_clip_iterations,
            self.enable_clip_interm_domain,
            self.clip_interm_topk,
            self.clip_in_alpha_crown,
            self.clip_interm_prune,
            self.clip_interm_use_final_layer,
            self.enable_interm_transfer,
        )
    }

    /// Create config optimized for speed (fewer iterations, simpler heuristic).
    #[staticmethod]
    pub(crate) fn fast() -> Self {
        BetaCrownConfig {
            max_domains: 100,
            timeout_secs: 30,
            max_depth: 10,
            use_alpha_crown: false,
            use_forward_bounds: false,
            use_crown_ibp: false,
            branching: PyBranchingHeuristic::LargestBoundWidth,
            fsb_candidates: 3,
            kfsb_reduce_op: PyKfsbReduceOp::Min,
            beta_lr: 0.1,
            beta_iterations: 10,
            beta_tolerance: 1e-4,
            alpha_lr: 0.1,
            batch_size: 8,
            enable_cuts: false,
            max_cuts: 100,
            enable_proactive_cuts: false,
            max_proactive_cuts: 50,
            enable_biccos_constraint_strengthening: false,
            biccos_drop_ratio: 0.5,
            enable_biccos_cold_start: false,
            biccos_min_verified: 5,
            biccos_min_verified_rate: 0.05,
            biccos_verified_rate_window: 20,
            biccos_min_cuts: 3,
            biccos_min_bound_gain: 1e-4,
            biccos_bound_gain_window: 20,
            biccos_cold_max_iters: 40,
            biccos_cut_window: 40,
            biccos_min_cut_yield: 0.05,
            biccos_cut_yield_window: 20,
            biccos_cut_yield_patience: 2,
            verify_upper_bound: false,
            enable_pgd_attack: false,
            pgd_restarts: 10,
            pgd_steps: 20,
            enable_relaxed_clip: false,
            relaxed_clip_iterations: 1,
            enable_interm_transfer: false,
            enable_clip_interm_domain: false,
            clip_interm_topk: 3,
            clip_in_alpha_crown: false,
            clip_interm_prune: false,
            clip_interm_use_final_layer: false,
        }
    }

    /// Create config optimized for precision (more iterations, smarter heuristic).
    #[staticmethod]
    pub(crate) fn precise() -> Self {
        BetaCrownConfig {
            max_domains: 10000,
            timeout_secs: 300,
            max_depth: 50,
            use_alpha_crown: true,
            use_forward_bounds: false,
            use_crown_ibp: true,
            branching: PyBranchingHeuristic::Kfsb,
            fsb_candidates: 16,
            kfsb_reduce_op: PyKfsbReduceOp::Min,
            beta_lr: 0.05,
            beta_iterations: 50,
            beta_tolerance: 1e-5,
            alpha_lr: 0.05,
            batch_size: 16,
            enable_cuts: true,
            max_cuts: 2000,
            enable_proactive_cuts: true,
            max_proactive_cuts: 200,
            enable_biccos_constraint_strengthening: true,
            biccos_drop_ratio: 0.5,
            enable_biccos_cold_start: false,
            biccos_min_verified: 5,
            biccos_min_verified_rate: 0.05,
            biccos_verified_rate_window: 20,
            biccos_min_cuts: 3,
            biccos_min_bound_gain: 1e-4,
            biccos_bound_gain_window: 20,
            biccos_cold_max_iters: 40,
            biccos_cut_window: 40,
            biccos_min_cut_yield: 0.05,
            biccos_cut_yield_window: 20,
            biccos_cut_yield_patience: 2,
            verify_upper_bound: false,
            enable_pgd_attack: true,
            pgd_restarts: 100,
            pgd_steps: 50,
            enable_relaxed_clip: false,
            relaxed_clip_iterations: 1,
            enable_interm_transfer: false,
            enable_clip_interm_domain: false,
            clip_interm_topk: 3,
            clip_in_alpha_crown: false,
            clip_interm_prune: false,
            clip_interm_use_final_layer: false,
        }
    }
}

impl BetaCrownConfig {
    /// Create from Rust config.
    pub(crate) fn from_rust(config: &RustBetaCrownConfig) -> Self {
        BetaCrownConfig {
            max_domains: config.max_domains,
            timeout_secs: config.timeout.as_secs(),
            max_depth: config.max_depth,
            use_alpha_crown: config.use_alpha_crown,
            use_forward_bounds: config.use_forward_bounds,
            use_crown_ibp: config.use_crown_ibp,
            branching: match config.branching_heuristic {
                BranchingHeuristic::LargestBoundWidth => PyBranchingHeuristic::LargestBoundWidth,
                BranchingHeuristic::BoundImpact => PyBranchingHeuristic::BoundImpact,
                BranchingHeuristic::FilteredSmartBranching => {
                    PyBranchingHeuristic::FilteredSmartBranching
                }
                BranchingHeuristic::Kfsb => PyBranchingHeuristic::Kfsb,
                BranchingHeuristic::KfsbInterceptOnly => PyBranchingHeuristic::KfsbInterceptOnly,
                BranchingHeuristic::Sequential => PyBranchingHeuristic::Sequential,
                BranchingHeuristic::InputSplit => PyBranchingHeuristic::InputSplit,
                BranchingHeuristic::GenBaB(_) => PyBranchingHeuristic::GenBaB,
            },
            fsb_candidates: config.fsb_candidates,
            kfsb_reduce_op: match config.kfsb_reduce_op {
                KfsbReduceOp::Min => PyKfsbReduceOp::Min,
                KfsbReduceOp::Max => PyKfsbReduceOp::Max,
                KfsbReduceOp::Mean => PyKfsbReduceOp::Mean,
            },
            beta_lr: config.beta_lr,
            beta_iterations: config.beta_iterations,
            beta_tolerance: config.beta_tolerance,
            alpha_lr: config.alpha_lr,
            batch_size: config.batch_size,
            enable_cuts: config.enable_cuts,
            max_cuts: config.max_cuts,
            enable_proactive_cuts: config.enable_proactive_cuts,
            max_proactive_cuts: config.max_proactive_cuts,
            enable_biccos_constraint_strengthening: config.enable_biccos_constraint_strengthening,
            biccos_drop_ratio: config.biccos_drop_ratio,
            enable_biccos_cold_start: config.enable_biccos_cold_start,
            biccos_min_verified: config.biccos_min_verified,
            biccos_min_verified_rate: config.biccos_min_verified_rate,
            biccos_verified_rate_window: config.biccos_verified_rate_window,
            biccos_min_cuts: config.biccos_min_cuts,
            biccos_min_bound_gain: config.biccos_min_bound_gain,
            biccos_bound_gain_window: config.biccos_bound_gain_window,
            biccos_cold_max_iters: config.biccos_cold_max_iters,
            biccos_cut_window: config.biccos_cut_window,
            biccos_min_cut_yield: config.biccos_min_cut_yield,
            biccos_cut_yield_window: config.biccos_cut_yield_window,
            biccos_cut_yield_patience: config.biccos_cut_yield_patience,
            verify_upper_bound: config.verify_upper_bound,
            enable_pgd_attack: config.enable_pgd_attack,
            pgd_restarts: config.pgd_restarts,
            pgd_steps: config.pgd_steps,
            enable_relaxed_clip: config.enable_relaxed_clip,
            relaxed_clip_iterations: config.relaxed_clip_iterations,
            enable_interm_transfer: config.enable_interm_transfer,
            enable_clip_interm_domain: config.enable_clip_interm_domain,
            clip_interm_topk: config.clip_interm_topk,
            clip_in_alpha_crown: config.clip_in_alpha_crown,
            clip_interm_prune: config.clip_interm_prune,
            clip_interm_use_final_layer: config.clip_interm_use_final_layer,
        }
    }

    /// Validate all float config fields are finite and non-negative.
    /// Returns a plain `String` error (no pyo3 dependency) for testability. (#3305)
    pub(crate) fn validate_inner(&self) -> Result<(), String> {
        let check = |name: &str, val: f32| -> Result<(), String> {
            if !val.is_finite() || val < 0.0 {
                return Err(format!("{name} must be finite and non-negative, got {val}"));
            }
            Ok(())
        };
        check("beta_lr", self.beta_lr)?;
        check("alpha_lr", self.alpha_lr)?;
        check("beta_tolerance", self.beta_tolerance)?;
        check("biccos_drop_ratio", self.biccos_drop_ratio)?;
        check("biccos_min_verified_rate", self.biccos_min_verified_rate)?;
        check("biccos_min_bound_gain", self.biccos_min_bound_gain)?;
        check("biccos_min_cut_yield", self.biccos_min_cut_yield)?;
        if self.use_alpha_crown && self.use_forward_bounds {
            return Err(
                "use_forward_bounds cannot be enabled together with use_alpha_crown".to_string(),
            );
        }
        Ok(())
    }

    /// Validate all float config fields, returning `PyValueError` on failure.
    ///
    /// Called at the `to_rust()` boundary to prevent NaN/Inf/negative values
    /// from flowing into the Rust verification engine. (#2899)
    pub(crate) fn validate(&self) -> PyResult<()> {
        self.validate_inner().map_err(PyValueError::new_err)
    }

    /// Convert to Rust config, validating all float fields first.
    ///
    /// Returns `PyValueError` if any float field is NaN, Inf, or negative.
    #[allow(clippy::field_reassign_with_default)] // 30+ fields — struct literal impractical
    pub(crate) fn to_rust(&self) -> PyResult<RustBetaCrownConfig> {
        self.validate()?;
        let mut config = RustBetaCrownConfig::default();
        config.max_domains = self.max_domains;
        config.timeout = std::time::Duration::from_secs(self.timeout_secs);
        config.max_depth = self.max_depth;
        config.use_alpha_crown = self.use_alpha_crown;
        config.use_forward_bounds = self.use_forward_bounds;
        config.use_crown_ibp = self.use_crown_ibp;
        config.branching_heuristic = self.branching.clone().into();
        config.fsb_candidates = self.fsb_candidates;
        config.kfsb_reduce_op = self.kfsb_reduce_op.clone().into();
        config.beta_lr = self.beta_lr;
        config.beta_iterations = self.beta_iterations;
        config.beta_tolerance = self.beta_tolerance;
        config.alpha_lr = self.alpha_lr;
        config.batch_size = self.batch_size;
        config.enable_cuts = self.enable_cuts;
        config.max_cuts = self.max_cuts;
        config.enable_proactive_cuts = self.enable_proactive_cuts;
        config.max_proactive_cuts = self.max_proactive_cuts;
        config.enable_biccos_constraint_strengthening = self.enable_biccos_constraint_strengthening;
        config.biccos_drop_ratio = self.biccos_drop_ratio;
        config.enable_biccos_cold_start = self.enable_biccos_cold_start;
        config.biccos_min_verified = self.biccos_min_verified;
        config.biccos_min_verified_rate = self.biccos_min_verified_rate;
        config.biccos_verified_rate_window = self.biccos_verified_rate_window;
        config.biccos_min_cuts = self.biccos_min_cuts;
        config.biccos_min_bound_gain = self.biccos_min_bound_gain;
        config.biccos_bound_gain_window = self.biccos_bound_gain_window;
        config.biccos_cold_max_iters = self.biccos_cold_max_iters;
        config.biccos_cut_window = self.biccos_cut_window;
        config.biccos_min_cut_yield = self.biccos_min_cut_yield;
        config.biccos_cut_yield_window = self.biccos_cut_yield_window;
        config.biccos_cut_yield_patience = self.biccos_cut_yield_patience;
        config.verify_upper_bound = self.verify_upper_bound;
        config.enable_pgd_attack = self.enable_pgd_attack;
        config.pgd_restarts = self.pgd_restarts;
        config.pgd_steps = self.pgd_steps;
        config.enable_relaxed_clip = self.enable_relaxed_clip;
        config.relaxed_clip_iterations = self.relaxed_clip_iterations;
        config.enable_interm_transfer = self.enable_interm_transfer;
        config.enable_clip_interm_domain = self.enable_clip_interm_domain;
        config.clip_interm_topk = self.clip_interm_topk;
        config.clip_in_alpha_crown = self.clip_in_alpha_crown;
        config.clip_interm_prune = self.clip_interm_prune;
        config.clip_interm_use_final_layer = self.clip_interm_use_final_layer;
        config
            .validate()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(config)
    }
}
