// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Configuration types for α-CROWN optimization.
//!
//! Contains `AlphaCrownConfig`, optimizer parameters (`AdamParams`, `Optimizer`),
//! intermediate storage types (`AlphaCrownIntermediate`, `GraphAlphaCrownIntermediate`),
//! and supporting enums (`GradientMethod`, `MultiSpecKeep`).
//!
//! Extracted from `alpha.rs` per #2201 to keep alpha state logic under 900 LOC.

use super::LinearBounds;
use crate::invprop::{InvpropConfig, OutputConstraints};
use ndarray::{Array1, Array2};
use ny_tensor::{next_down_f32, next_up_f32};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Spec-proven early-exit for the single-objective α-CROWN warmup loop.
///
/// When present on an `AlphaCrownConfig`, the iterative warmup loop projects its
/// per-iteration output bounds onto `objective` (interval arithmetic) and stops
/// optimizing the moment the resulting scalar `(lower, upper)` already proves the
/// property against `threshold` — there is no point spending the remaining
/// iterations once the bound has cleared the decision threshold (#warmup-early-exit).
///
/// SOUNDNESS: this is a STRICT optimization-stop only. The bound at the exit
/// iteration is already a valid over-approximation that clears `threshold`; the
/// loop returns the best valid bound it has found. No bound *computation* changes.
/// When this is `None` (every non-warmup caller) the loop runs exactly as before.
#[derive(Debug, Clone, PartialEq)]
pub struct AlphaSpecEarlyExit {
    /// Linear objective vector `c`; the scalar bound is `c^T y` over the output interval.
    pub objective: Vec<f32>,
    /// Decision threshold the projected bound must clear to prove the property.
    pub threshold: f32,
    /// Verification mode (matches `BetaCrownConfig::verify_upper_bound`):
    /// true → property proven when `upper < threshold`; false → when `lower > threshold`.
    pub verify_upper_bound: bool,
}

impl AlphaSpecEarlyExit {
    /// Interval-arithmetic projection of an output `BoundedTensor` onto `self.objective`,
    /// returning the scalar `(lower, upper)` bound on `c^T y`. Mirrors
    /// `beta_crown::engine::graph::objectives::objective_bounds` exactly so the
    /// in-loop early-exit check sees the same projection the post-warmup code uses.
    /// Returns `None` on a length mismatch (caller then skips the early-exit check
    /// for that iteration, preserving the no-change-on-mismatch contract).
    ///
    /// SOUNDNESS (#concretize-soundness-hardening): this projection feeds
    /// `is_verified`, a verdict early-exit, where an inward round-to-nearest
    /// endpoint would be an undetectable false Verified. Accumulate in f64 —
    /// `f32 x f32` promoted to f64 is exact (48 < 53 significand bits), so only
    /// the additions round — then close with a directed f32 cast, mirroring
    /// `LinearBounds::concretize_sound`.
    #[must_use]
    pub fn project_bounds(&self, lower: &[f32], upper: &[f32]) -> Option<(f32, f32)> {
        if lower.len() != self.objective.len() || upper.len() != self.objective.len() {
            return None;
        }
        let mut lo = 0.0f64;
        let mut hi = 0.0f64;
        for (idx, &c) in self.objective.iter().enumerate() {
            let c = c as f64;
            let l = lower[idx] as f64;
            let u = upper[idx] as f64;
            if c >= 0.0 {
                lo += c * l;
                hi += c * u;
            } else {
                lo += c * u;
                hi += c * l;
            }
        }
        // Guard against NaN from degenerate CROWN propagation (#2359 parity with
        // `objective_bounds`): collapse to the sound unbounded interval, which
        // `is_verified` rejects — the loop simply keeps optimizing.
        if lo.is_nan() || hi.is_nan() {
            return Some((f32::NEG_INFINITY, f32::INFINITY));
        }
        Some((next_down_f32(lo as f32), next_up_f32(hi as f32)))
    }

    /// True when the projected scalar bounds already prove the property. Mirrors
    /// `BetaCrownConfig::domain_is_verified_for_mode` exactly (rejects all
    /// non-finite inputs; #2993). Used by the warmup loop to break early.
    #[must_use]
    pub fn is_verified(&self, lower: f32, upper: f32) -> bool {
        if !lower.is_finite() || !upper.is_finite() || !self.threshold.is_finite() {
            return false;
        }
        if self.verify_upper_bound {
            upper < self.threshold
        } else {
            lower > self.threshold
        }
    }
}

/// Gradient estimation method for α-CROWN optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GradientMethod {
    /// Finite differences: perturb each α individually (O(n) passes per iteration).
    /// Accurate but slow for large networks.
    FiniteDifferences,
    /// SPSA: Simultaneous Perturbation Stochastic Approximation.
    /// Perturbs all α at once with random directions (O(1) passes per iteration).
    /// Faster but noisier — with 1 sample, gradient variance is O(n) per parameter,
    /// making it ineffective for networks with many unstable neurons.
    /// Use AnalyticChain instead for production.
    Spsa,
    /// Analytic gradients: compute local gradients during CROWN backward pass.
    /// O(1) passes per iteration, approximate (local, not chain-rule) gradients.
    ///
    /// **EXPERIMENTAL**: This computes local gradients at each ReLU layer but doesn't
    /// properly chain them through subsequent layers to the input. The gradient represents
    /// coefficient sensitivity, not the actual bound gradient. Use AnalyticChain instead
    /// for production.
    Analytic,
    /// True analytic gradients with full chain-rule propagation.
    /// Computes ∂(output_lower)/∂α_i by propagating gradients through all downstream layers.
    ///
    /// Stores intermediate A matrices at each ReLU during backward pass, then computes
    /// true chain-rule gradients: for neuron i in ReLU layer k, gradient is
    /// Σ_j A_downstream\[j,i\] × input_contribution\[i\] where j sums over output dimensions.
    ///
    /// This is the closest Rust equivalent to PyTorch's `loss.backward()` used in the
    /// reference α,β-CROWN implementation (`optimized_bounds.py:870`). SPSA with 1 sample
    /// produces noise-dominated gradient estimates for networks with many unstable neurons,
    /// whereas AnalyticChain computes true chain-rule gradients in O(1) backward passes.
    #[default]
    AnalyticChain,
}

/// Intermediate values stored during α-CROWN backward pass for chain-rule gradient computation.
///
/// For chain-rule gradients, we need to know the A matrix (linear bounds coefficients) at
/// each ReLU layer BEFORE the ReLU is applied. This struct stores these values.
#[derive(Debug, Clone)]
pub struct AlphaCrownIntermediate {
    /// A matrices at each ReLU layer (before ReLU applied), in forward layer order.
    /// a_at_relu\[k\] is the A matrix from output back to just before ReLU layer k.
    /// Shape of each: (num_outputs, num_neurons_at_relu_k)
    pub a_at_relu: Vec<Array2<f32>>,

    /// Pre-ReLU bounds at each ReLU layer (for determining unstable neurons).
    /// Shape: (num_relu_layers,) where each element has shape (num_neurons,)
    pub pre_relu_bounds: Vec<(Array1<f32>, Array1<f32>)>,

    /// Final linear bounds after complete backward pass.
    pub final_bounds: LinearBounds,
}

impl AlphaCrownIntermediate {
    /// Create empty intermediate storage.
    pub fn new() -> Self {
        Self {
            a_at_relu: Vec::new(),
            pre_relu_bounds: Vec::new(),
            final_bounds: LinearBounds::identity(1),
        }
    }
}

impl Default for AlphaCrownIntermediate {
    fn default() -> Self {
        Self::new()
    }
}

/// Intermediate values stored during DAG α-CROWN backward pass for chain-rule gradient computation.
///
/// Unlike `AlphaCrownIntermediate` which uses Vec for sequential networks, this uses HashMap
/// to support DAG structures where ReLU nodes are identified by name.
#[derive(Debug, Clone)]
pub struct GraphAlphaCrownIntermediate {
    /// A matrices at each ReLU node (before ReLU applied), keyed by node name.
    /// Each entry is the accumulated A matrix from output back to just before that ReLU node.
    /// Shape of each: (num_outputs, num_neurons_at_relu)
    pub a_at_relu: std::collections::HashMap<String, Array2<f32>>,

    /// Pre-ReLU bounds at each ReLU node (for determining unstable neurons).
    /// Shape: each entry is (lower, upper) arrays with shape (num_neurons,)
    pub pre_relu_bounds: std::collections::HashMap<String, (Array1<f32>, Array1<f32>)>,

    /// Final linear bounds (accumulated to input).
    pub final_bounds: LinearBounds,

    /// Alpha gradients at each ReLU node, keyed by node name.
    /// Each entry is ∂(lower_bound)/∂α for each neuron at that ReLU.
    /// Populated when alpha state is provided during the backward pass.
    /// Issue: #1841
    pub alpha_gradients: std::collections::HashMap<String, Array1<f32>>,

    /// Upper-path alpha gradients at each ReLU node, keyed by node name.
    /// Each entry is ∂(upper_bound)/∂α_upper for each neuron at that ReLU.
    /// Populated when dual-alpha state is provided during the backward pass.
    pub alpha_gradients_upper: std::collections::HashMap<String, Array1<f32>>,
}

impl GraphAlphaCrownIntermediate {
    /// Create empty intermediate storage.
    pub fn new() -> Self {
        Self {
            a_at_relu: std::collections::HashMap::new(),
            pre_relu_bounds: std::collections::HashMap::new(),
            final_bounds: LinearBounds::identity(1),
            alpha_gradients: std::collections::HashMap::new(),
            alpha_gradients_upper: std::collections::HashMap::new(),
        }
    }

    /// The A matrix at a specific ReLU node.
    pub fn a_at_relu(&self, node_name: &str) -> Option<&Array2<f32>> {
        self.a_at_relu.get(node_name)
    }

    /// Pre-ReLU bounds at a specific ReLU node.
    pub fn pre_relu_bounds(&self, node_name: &str) -> Option<&(Array1<f32>, Array1<f32>)> {
        self.pre_relu_bounds.get(node_name)
    }
}

impl Default for GraphAlphaCrownIntermediate {
    fn default() -> Self {
        Self::new()
    }
}

/// Adam optimizer hyperparameters.
///
/// Bundles Adam-specific parameters to reduce function argument counts.
/// Matches PyTorch/auto_LiRPA defaults: β₁=0.9, β₂=0.999, ε=1e-8.
#[derive(Debug, Clone, Copy)]
pub struct AdamParams {
    /// Learning rate (step size)
    pub learning_rate: f32,
    /// Exponential decay rate for first moment (β₁), default: 0.9
    pub beta1: f32,
    /// Exponential decay rate for second moment (β₂), default: 0.999
    pub beta2: f32,
    /// Small constant for numerical stability (ε), default: 1e-8
    pub epsilon: f32,
    /// Iteration number (1-indexed for bias correction)
    pub t: usize,
}

impl AdamParams {
    /// Create new Adam parameters with default hyperparameters.
    ///
    /// Uses PyTorch/auto_LiRPA defaults: β₁=0.9, β₂=0.999, ε=1e-8.
    pub fn new(learning_rate: f32, t: usize) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t,
        }
    }

    /// Create Adam parameters with custom hyperparameters.
    pub fn with_hyperparams(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        t: usize,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t,
        }
    }
}

/// Optimizer type for alpha parameter updates.
///
/// Ported from α,β-CROWN's proven configurations:
/// - α,β-CROWN default: Adam with lr=0.1, beta1=0.9, beta2=0.999
/// - This dramatically outperforms SGD+momentum for bound tightening.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Optimizer {
    /// SGD with momentum. Original ny default.
    Sgd,
    /// Adam optimizer (adaptive moment estimation).
    /// Default: matches α,β-CROWN's proven configuration.
    #[default]
    Adam,
}

/// Multi-spec keep/prune behavior during bound optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MultiSpecKeep {
    /// Keep all specification bounds (auto_LiRPA default).
    #[default]
    All,
}

/// Serde default for `early_stop_patience` field (backward compat with configs
/// serialized before #3298).
fn default_early_stop_patience() -> usize {
    10
}

/// Serde default for `start_save_best`. Matches α,β-CROWN default (0.5).
fn default_start_save_best() -> f32 {
    0.5
}

/// Serde default for `full_conv_alpha`. Default true preserves current behavior
/// (per-neuron alpha for conv layers). Setting false enables channel-shared alpha
/// which uses dramatically fewer parameters for large conv layers (cifar100).
/// Source: alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/arguments.py:348.
fn default_full_conv_alpha() -> bool {
    true
}

/// Configuration for alpha-CROWN optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaCrownConfig {
    /// Number of optimization iterations.
    pub iterations: usize,
    /// Learning rate for alpha parameter updates.
    pub learning_rate: f32,
    /// Learning rate decay factor per iteration.
    pub lr_decay: f32,
    /// Early stop if improvement is below this threshold.
    pub tolerance: f32,
    /// Whether to use adaptive learning rate (momentum-like).
    pub use_momentum: bool,
    /// Momentum coefficient (if use_momentum is true).
    pub momentum: f32,
    /// Gradient estimation method.
    pub gradient_method: GradientMethod,
    /// Number of SPSA samples to average per iteration (reduces variance).
    pub spsa_samples: usize,
    /// Use IBP bounds for intermediate nodes (O(N)) instead of CROWN-IBP (O(N²)).
    ///
    /// When true (default), matches α,β-CROWN's `fix_interm_bounds=True` behavior:
    /// - Use cheap IBP bounds for intermediate node bounds
    /// - Only run CROWN backward pass for output node optimization
    ///
    /// This dramatically speeds up initialization for deep networks (10x+ for ResNet-4b).
    /// Set to false for tighter intermediate bounds at the cost of O(N²) initialization.
    pub fix_interm_bounds: bool,
    /// Sparse alpha optimization: only optimize the top K% most influential alphas.
    ///
    /// After the first iteration, alphas are ranked by gradient magnitude and only
    /// the top `sparse_ratio` fraction are optimized in subsequent iterations.
    /// This reduces SPSA variance and focuses optimization where it matters most.
    ///
    /// Set to 1.0 to disable sparsity (optimize all alphas).
    /// Recommended: 0.1-0.3 for deep networks with many unstable neurons.
    pub sparse_ratio: f32,
    /// Adaptive skip: optional ny heuristic that bypasses α-CROWN by depth.
    ///
    /// When enabled:
    /// - Networks with more than `adaptive_skip_depth_threshold` ReLU layers skip α-CROWN
    /// - Optionally runs a 1-iteration pilot before committing to the skip decision
    ///
    /// Default: false (disabled). The reference α,β-CROWN has no depth gate — it relies
    /// on patience-based early stopping (`early_stop_patience`) to terminate when
    /// optimization stalls. With the threshold at 20, adaptive skip silently prevented
    /// α-CROWN from running on production graph models (HTDemucs, Whisper, Kokoro, etc.),
    /// causing CROWN bounds identical to IBP (#3918).
    pub adaptive_skip: bool,
    /// Depth threshold for adaptive skipping.
    ///
    /// Networks with more than this many ReLU layers will skip α-CROWN optimization
    /// if `adaptive_skip` is enabled. This threshold is retained for explicit opt-in
    /// uses of the heuristic even though the default configuration disables it.
    ///
    /// Default: 20 (raised from 8 for medium-depth DAGs, #3619)
    pub adaptive_skip_depth_threshold: usize,
    /// Run a pilot iteration to check if α-CROWN helps before full optimization.
    ///
    /// When enabled with `adaptive_skip`, runs 1 iteration and compares the improvement
    /// to CROWN bounds. If improvement is below `pilot_improvement_threshold`, skips
    /// remaining iterations.
    ///
    /// This catches cases where depth isn't the only factor (e.g., already tight bounds).
    /// Default: false (#3298 — pilot exits too early for deep DAG networks)
    pub adaptive_skip_pilot: bool,
    /// Minimum improvement required from pilot iteration to continue optimization.
    ///
    /// If the first iteration improves lower bound sum by less than this amount,
    /// skip remaining iterations. The value is absolute (not relative).
    ///
    /// Default: 1e-3 (skip if pilot improvement < 0.001)
    pub pilot_improvement_threshold: f32,
    /// Number of consecutive non-improving iterations before early stopping.
    ///
    /// Matches α,β-CROWN's `early_stop_patience` parameter (auto_LiRPA
    /// `optimized_bounds.py:75-77`, default 10). Previously hardcoded to 3, which
    /// was too aggressive for deep DAG networks where gradient attenuation causes
    /// slow convergence.
    ///
    /// Default: 10 (matches α,β-CROWN reference)
    #[serde(default = "default_early_stop_patience")]
    pub early_stop_patience: usize,
    /// Optimizer to use for alpha parameter updates.
    ///
    /// Ported from α,β-CROWN: Adam significantly outperforms SGD for bound tightening.
    /// Default: Adam (matches α,β-CROWN default)
    pub optimizer: Optimizer,
    /// Adam β₁: exponential decay rate for first moment estimate.
    /// Default: 0.9 (matches α,β-CROWN and PyTorch default)
    pub adam_beta1: f32,
    /// Adam β₂: exponential decay rate for second moment estimate.
    /// Default: 0.999 (matches α,β-CROWN and PyTorch default)
    pub adam_beta2: f32,
    /// Adam ε: small constant for numerical stability.
    /// Default: 1e-8 (matches α,β-CROWN and PyTorch default)
    pub adam_epsilon: f32,

    /// In-iteration pruning of verified domains (auto_LiRPA's
    /// `pruning_in_iteration` toggle).
    ///
    /// NOT YET IMPLEMENTED: declared for reference-config compatibility, but no
    /// optimization loop reads it — setting it has no effect. The preset loader
    /// warns and ignores the corresponding `bab.pruning_in_iteration` key.
    /// Default: false.
    #[serde(default)]
    pub pruning_in_iteration: bool,
    /// Minimum fraction of domains eligible for in-iteration pruning
    /// (auto_LiRPA's `pruning_in_iteration_threshold`).
    ///
    /// NOT YET IMPLEMENTED: declared for reference-config compatibility, but no
    /// optimization loop reads it — setting it has no effect.
    /// Default: 0.2.
    #[serde(default)]
    pub pruning_in_iteration_threshold: f32,
    /// Multi-spec keep/prune hook for bound optimization.
    ///
    /// Matches auto_LiRPA's `multi_spec_keep_func` control.
    /// Default: `MultiSpecKeep::All`.
    #[serde(default)]
    pub multi_spec_keep_func: MultiSpecKeep,

    /// INVPROP configuration for output constraint backward propagation.
    ///
    /// When enabled, output constraints (A·y <= rhs) are propagated backward
    /// to tighten intermediate bounds before BaB.
    ///
    /// Default: disabled (InvpropConfig::default())
    #[serde(default)]
    pub invprop: InvpropConfig,

    /// Output constraints for INVPROP optimization.
    ///
    /// When set and `invprop.enabled` is true, these constraints are used
    /// to initialize ny dual variables and optimize them alongside alphas.
    /// This implements the incomplete verifier integration for INVPROP (#371).
    ///
    /// Default: None (no output constraints)
    #[serde(skip)]
    pub output_constraints: Option<OutputConstraints>,

    /// Fraction of optimization iterations to skip before saving best bounds.
    ///
    /// Matches α,β-CROWN's `start_save_best` parameter (auto_LiRPA
    /// `optimized_bounds.py:80`, default 0.5). Best bounds are saved at
    /// iteration 0 (baseline), then skipped for iterations 1 through
    /// `total_iterations * start_save_best`, then saved every iteration after.
    ///
    /// Rationale: early alpha/beta values are random noise; element-wise max
    /// over noisy intermediate iterations can lock in suboptimal per-element
    /// bounds that never improve.
    ///
    /// Default: 0.5 (skip first 50% of iterations)
    #[serde(default = "default_start_save_best")]
    pub start_save_best: f32,

    /// Use per-neuron alpha for conv layers (full_conv_alpha).
    ///
    /// When true (default), every spatial position in a conv layer gets its own
    /// alpha parameter. When false, alphas are shared across the spatial
    /// (channel) dimension, dramatically reducing the number of alpha parameters
    /// for large conv layers. The reference α,β-CROWN cifar100 config sets this
    /// to false — without it, conv alpha count is ~63x too high (#4404).
    ///
    /// Default: true (preserves current per-neuron behavior until channel-shared
    /// layout is activated in later slices).
    /// Source: alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/arguments.py:348.
    #[serde(default = "default_full_conv_alpha")]
    pub full_conv_alpha: bool,

    /// Wall-clock deadline for alpha optimization (#2698).
    ///
    /// When set, the optimization loop checks this deadline at the start of each
    /// iteration and returns current best bounds if exceeded. This prevents
    /// alpha-CROWN from consuming the entire verification timeout budget during
    /// initial bound computation for large models.
    ///
    /// Default: None (no deadline, run all iterations)
    #[serde(skip)]
    pub deadline: Option<Instant>,

    /// Optional spec-proven early-exit for the single-objective warmup loop
    /// (#warmup-early-exit). When `Some`, the iterative α-CROWN warmup loop projects
    /// its per-iteration output bounds onto the objective and stops the moment the
    /// projected bound already proves the property against the threshold. SOUND:
    /// stops optimizing sooner only; the bound returned is the best already-valid
    /// over-approximation. `None` (every non-warmup caller) → no behavior change.
    #[serde(skip)]
    pub spec_early_exit: Option<AlphaSpecEarlyExit>,
}

impl Default for AlphaCrownConfig {
    fn default() -> Self {
        Self {
            // α,β-CROWN default: 100 iterations for incomplete verifier.
            // Early stop patience (10) terminates early when bounds converge.
            // Source: alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/arguments.py:354 (init_iteration=100).
            iterations: 100,
            // α,β-CROWN default: lr_init_alpha=0.1.
            // Source: alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/arguments.py:351.
            learning_rate: 0.1,
            lr_decay: 0.98,
            tolerance: 1e-4,
            use_momentum: true,
            momentum: 0.9,
            // AnalyticChain: true chain-rule gradients, closest to reference α,β-CROWN's
            // loss.backward() (optimized_bounds.py:870). SPSA with 1 sample is noise-dominated
            // for networks with many unstable neurons (#2035).
            gradient_method: GradientMethod::AnalyticChain,
            spsa_samples: 1, // Only used when gradient_method is Spsa
            // Default to true: use IBP bounds for intermediates (fast O(N)).
            // Matches α,β-CROWN's fix_interm_bounds=True default.
            fix_interm_bounds: true,
            // Sparse optimization: focus on top 30% most influential alphas.
            // Reduces SPSA variance when perturbing fewer coordinates.
            sparse_ratio: 0.3,
            // Adaptive skip disabled by default (#3918): the upstream reference has
            // no depth gate — it relies on early_stop_patience to terminate stalled
            // optimization. The threshold of 20 silently prevented α-CROWN from running
            // on all production graph models (>20 ReLU nodes), producing IBP-identical
            // bounds. The retained threshold (20) only applies when explicitly enabled.
            adaptive_skip: false,
            adaptive_skip_depth_threshold: 20,
            // Pilot iteration disabled by default (#3298): the 1e-3 absolute threshold
            // exits too early for deep DAG networks (ResNet-2b) where gradient attenuation
            // makes first-iteration improvement tiny. The reference α,β-CROWN has no pilot
            // concept — it uses patience-based early stopping instead (early_stop_patience).
            adaptive_skip_pilot: false,
            // Require at least 1e-3 improvement from pilot to continue optimization.
            pilot_improvement_threshold: 1e-3,
            // Patience: 10 consecutive non-improving iterations before stopping.
            // Matches α,β-CROWN's early_stop_patience=10 (optimized_bounds.py:75-77).
            // Previously hardcoded to 3 — too aggressive for deep DAG networks (#3298).
            early_stop_patience: 10,
            // Optimizer: Adam (ported from α,β-CROWN's proven configuration)
            optimizer: Optimizer::Adam,
            // Adam hyperparameters (match α,β-CROWN and PyTorch defaults)
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
            // In-iteration pruning off by default (and not yet implemented — see field docs)
            pruning_in_iteration: false,
            pruning_in_iteration_threshold: 0.2,
            // Multi-spec pruning defaults to keeping all bounds
            multi_spec_keep_func: MultiSpecKeep::All,
            // INVPROP disabled by default
            invprop: InvpropConfig::default(),
            // No output constraints by default (set programmatically when needed)
            output_constraints: None,
            // Skip saving best bounds during first 50% of iterations.
            // Matches α,β-CROWN's start_save_best=0.5 (optimized_bounds.py:80).
            start_save_best: 0.5,
            // Per-neuron alpha for conv layers (full_conv_alpha=True in reference).
            // Setting false enables channel-shared alpha (63x fewer params for cifar100).
            // Source: alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/arguments.py:348.
            full_conv_alpha: true,
            // No deadline by default (run all iterations)
            deadline: None,
            // No spec early-exit by default: warmup runs to the iteration/time cap
            // exactly as before. Set programmatically by the single-objective ReLU-split
            // warmup so it can stop once the root bound clears the threshold.
            spec_early_exit: None,
        }
    }
}

impl AlphaCrownConfig {
    /// Whether best bounds should be saved at the given iteration.
    ///
    /// Matches α,β-CROWN's `start_save_best` skip window (auto_LiRPA
    /// `optimized_bounds.py:785-797`): save at iteration 0 (baseline), skip
    /// iterations 1 through `total_iterations * start_save_best`, then save
    /// every iteration after the warmup window.
    ///
    /// When `force` is true, always save regardless of iteration (used for
    /// early-exit conditions like patience exhaustion, stop criterion, or
    /// deadline — matches reference lines 793-795).
    pub fn should_save_best(&self, iter: usize, force: bool) -> bool {
        if force {
            return true;
        }
        let threshold = (self.iterations as f32 * self.start_save_best) as usize;
        iter == 0 || iter > threshold
    }

    /// Create Adam parameters from this config for the given learning rate and iteration.
    ///
    /// This bundles the Adam hyperparameters (β₁, β₂, ε) from the config with the
    /// current learning rate (after decay) and iteration number.
    pub fn adam_params(&self, learning_rate: f32, t: usize) -> AdamParams {
        AdamParams::with_hyperparams(
            learning_rate,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
            t,
        )
    }

    /// Check whether the deadline has been exceeded (#2698).
    ///
    /// Returns `true` if a deadline is set and the current time is past it.
    pub fn past_deadline(&self) -> bool {
        self.deadline.map(|d| Instant::now() >= d).unwrap_or(false)
    }
}

#[cfg(test)]
#[path = "alpha_config_tests.rs"]
mod tests;
