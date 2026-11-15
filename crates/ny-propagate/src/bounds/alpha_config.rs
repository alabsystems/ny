// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Configuration types for α-CROWN optimization.
//!
//! Contains `AlphaCrownConfig`, optimizer parameters (`AdamParams`, `Optimizer`),
//! intermediate storage types (`AlphaCrownIntermediate`, `GraphAlphaCrownIntermediate`),
//! and supporting enums (`GradientMethod`, `MultiSpecKeep`).
//!
//! Extracted from the alpha module to keep alpha state logic under 900 LOC.

use super::LinearBounds;
use crate::invprop::{InvpropConfig, OutputConstraints};
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use std::{mem::size_of, time::Instant};

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
    /// Reuse the certified double-double spec-row reducer (with its directed
    /// f64 fallback) so this early-exit is bit-identical to the post-warmup
    /// root projection and remains useful under catastrophic cancellation.
    #[must_use]
    pub fn project_bounds(&self, lower: &[f32], upper: &[f32]) -> Option<(f32, f32)> {
        if lower.len() != self.objective.len() || upper.len() != self.objective.len() {
            return None;
        }
        let lo = super::certified_affine_sum_f32(
            0.0,
            self.objective
                .iter()
                .enumerate()
                .map(|(index, &coefficient)| {
                    let endpoint = if coefficient >= 0.0 {
                        lower[index]
                    } else {
                        upper[index]
                    };
                    (coefficient, endpoint)
                }),
            super::OutwardDirection::Lower,
        );
        let hi = super::certified_affine_sum_f32(
            0.0,
            self.objective
                .iter()
                .enumerate()
                .map(|(index, &coefficient)| {
                    let endpoint = if coefficient >= 0.0 {
                        upper[index]
                    } else {
                        lower[index]
                    };
                    (coefficient, endpoint)
                }),
            super::OutwardDirection::Upper,
        );
        Some((ny_core::f64_to_f32_down(lo), ny_core::f64_to_f32_up(hi)))
    }

    /// True when the projected scalar bounds already prove the property. Mirrors
    /// `BetaCrownConfig::domain_is_verified_for_mode` exactly (rejects all
    /// non-finite inputs). Used by the warmup loop to break early.
    #[must_use]
    pub fn is_verified(&self, lower: f32, upper: f32) -> bool {
        if !lower.is_finite() || !upper.is_finite() || lower > upper || !self.threshold.is_finite()
        {
            return false;
        }
        if self.verify_upper_bound {
            upper < self.threshold
        } else {
            lower > self.threshold
        }
    }

    /// Signed proven-margin slack of this row under the given output box.
    ///
    /// `> 0` ⇔ [`Self::is_verified`] holds; the magnitude says how far past (or
    /// short of) the threshold the projection lands. Returns `None` on a length
    /// mismatch or a non-finite projection so callers fail closed.
    #[must_use]
    pub fn margin_slack(&self, lower: &[f32], upper: &[f32]) -> Option<f32> {
        let (lo, hi) = self.project_bounds(lower, upper)?;
        let slack = if self.verify_upper_bound {
            self.threshold - hi
        } else {
            lo - self.threshold
        };
        slack.is_finite().then_some(slack)
    }
}

/// Multi-row spec objective used to RANK α iterates in the root warmup
/// (#root-alpha-margin).
///
/// ## Why this exists
///
/// The root warmup ascends `finite_lower_sum` — a plain sum over the RAW output
/// dimensions — while multi-class properties are a conjunction of MARGIN rows
/// (`y_true - y_i >= 0`). Worse, the loop returns its *last* α iterate, which is
/// one optimizer update ahead of the last bound it evaluated, and there is no
/// best-α snapshot. So iterations spent ascending the wrong objective hand the
/// downstream spec pass an α that can be strictly worse for the margins than the
/// one it started from, with no path back.
///
/// ## SOUNDNESS — this is SELECTION ONLY
///
/// [`Self::hinge_score`] is a *ranking* scalar. It never decides a verdict, never
/// claims a proof, and never feeds a bound. It only chooses WHICH α to keep, and
/// every α ∈ [0,1] yields a valid bound — so a badly-ranked α can cost bound
/// quality but can never produce a wrong verdict. The verdict-grade projection
/// remains [`AlphaSpecEarlyExit::is_verified`].
#[derive(Debug, Clone, PartialEq)]
pub struct AlphaSpecAscent {
    /// One row per property row. Each carries its own objective, threshold, and
    /// direction, so mixed-direction conjunctions are handled per row.
    pub rows: Vec<AlphaSpecEarlyExit>,
}

impl AlphaSpecAscent {
    /// Fail-closed constructor: `None` unless there is at least one row and every
    /// row shares one non-zero objective width.
    #[must_use]
    pub fn new(rows: Vec<AlphaSpecEarlyExit>) -> Option<Self> {
        let width = rows.first()?.objective.len();
        if width == 0 || rows.iter().any(|r| r.objective.len() != width) {
            return None;
        }
        Some(Self { rows })
    }

    /// The objective width every row shares.
    #[must_use]
    pub fn output_len(&self) -> usize {
        self.rows.first().map_or(0, |r| r.objective.len())
    }

    /// Hinge score `Σ_r min(0, slack_r)`: always `<= 0`, and `== 0` exactly when
    /// every row already clears its threshold.
    ///
    /// Only UNPROVEN rows contribute, so ascent effort is never spent padding a
    /// margin that is already comfortably positive — which is precisely the
    /// failure mode of summing raw logits. Higher (closer to zero) is better.
    ///
    /// Returns `None` if any row fails to project, so the caller keeps its current
    /// best rather than ranking on a partial score.
    #[must_use]
    pub fn hinge_score(&self, lower: &[f32], upper: &[f32]) -> Option<f32> {
        let mut acc = 0.0f64;
        for row in &self.rows {
            acc += f64::from(row.margin_slack(lower, upper)?).min(0.0);
        }
        let score = acc as f32;
        score.is_finite().then_some(score)
    }

    /// Count of rows already clearing their threshold. Telemetry only.
    #[must_use]
    pub fn verified_rows(&self, lower: &[f32], upper: &[f32]) -> usize {
        self.rows
            .iter()
            .filter(|r| {
                r.project_bounds(lower, upper)
                    .is_some_and(|(lo, hi)| r.is_verified(lo, hi))
            })
            .count()
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

/// Compact lower-A columns used only by analytical beta gradients.
#[derive(Debug, Clone)]
struct GraphBetaSparseA {
    /// Global input-flat neuron indices, strictly ascending and duplicate-free.
    neuron_indices: Vec<usize>,
    /// One compact column per `neuron_indices` entry.
    values: Array2<f32>,
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
    pub alpha_gradients: std::collections::HashMap<String, Array1<f32>>,

    /// Upper-path alpha gradients at each ReLU node, keyed by node name.
    /// Each entry is ∂(upper_bound)/∂α_upper for each neuron at that ReLU.
    /// Populated when dual-alpha state is provided during the backward pass.
    pub alpha_gradients_upper: std::collections::HashMap<String, Array1<f32>>,

    /// Beta-only compact lower-A columns captured under the default-dark sparse
    /// Patches gate. Kept private so general graph-alpha consumers cannot mistake
    /// a partial matrix for the historical full `a_at_relu` relation.
    beta_sparse_a_at_relu: std::collections::HashMap<String, GraphBetaSparseA>,

    /// #vit-margin-alpha (B4): the incoming LOWER coefficient rows at every
    /// α-bearing bilinear node (`BilinearCrown` or var×var `MatMul`), keyed by
    /// node name, shape `(num_outputs, num_bilinear_outputs)`. Populated ONLY
    /// by the spec-row exact-seed walk of the certified margin ascent, which
    /// differentiates its interpolated McCormick planes against these rows;
    /// every other walk leaves it empty, so no existing caller changes.
    pub(crate) a_at_bilinear: std::collections::HashMap<String, Array2<f32>>,
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
            beta_sparse_a_at_relu: std::collections::HashMap::new(),
            a_at_bilinear: std::collections::HashMap::new(),
        }
    }

    /// Logical heap payload retained by this intermediate proof object.
    ///
    /// This includes every numeric buffer, sparse beta index, map key, and the
    /// final linear relation. Hash-table buckets and allocator capacity slack
    /// are intentionally excluded because standard containers do not expose a
    /// complete resident-byte census; the global CROWN budget leaves seven
    /// eighths of process headroom for that overhead. Deadline-aware staging
    /// APIs use this value as their already-live request payload.
    pub(crate) fn logical_memory_bytes(&self) -> usize {
        let mut bytes = self.final_bounds.memory_bytes();
        for (key, values) in &self.a_at_relu {
            bytes = bytes
                .saturating_add(key.len())
                .saturating_add(values.len().saturating_mul(size_of::<f32>()));
        }
        for (key, (lower, upper)) in &self.pre_relu_bounds {
            bytes = bytes
                .saturating_add(key.len())
                .saturating_add(lower.len().saturating_mul(size_of::<f32>()))
                .saturating_add(upper.len().saturating_mul(size_of::<f32>()));
        }
        for (key, values) in &self.alpha_gradients {
            bytes = bytes
                .saturating_add(key.len())
                .saturating_add(values.len().saturating_mul(size_of::<f32>()));
        }
        for (key, values) in &self.alpha_gradients_upper {
            bytes = bytes
                .saturating_add(key.len())
                .saturating_add(values.len().saturating_mul(size_of::<f32>()));
        }
        for (key, sparse) in &self.beta_sparse_a_at_relu {
            bytes = bytes
                .saturating_add(key.len())
                .saturating_add(
                    sparse
                        .neuron_indices
                        .len()
                        .saturating_mul(size_of::<usize>()),
                )
                .saturating_add(sparse.values.len().saturating_mul(size_of::<f32>()));
        }
        for (key, values) in &self.a_at_bilinear {
            bytes = bytes
                .saturating_add(key.len())
                .saturating_add(values.len().saturating_mul(size_of::<f32>()));
        }
        bytes
    }

    /// The A matrix at a specific ReLU node.
    pub fn a_at_relu(&self, node_name: &str) -> Option<&Array2<f32>> {
        self.a_at_relu.get(node_name)
    }

    /// Pre-ReLU bounds at a specific ReLU node.
    pub fn pre_relu_bounds(&self, node_name: &str) -> Option<&(Array1<f32>, Array1<f32>)> {
        self.pre_relu_bounds.get(node_name)
    }

    /// Store selected lower-A columns for analytical beta sensitivity.
    ///
    /// This crate-private entry point deliberately cannot populate the public
    /// full-matrix map. Callers must provide canonical sorted global indices and
    /// a matching compact matrix.
    pub(crate) fn insert_beta_sparse_a(
        &mut self,
        node_name: String,
        neuron_indices: Vec<usize>,
        values: Array2<f32>,
    ) -> bool {
        if values.ncols() != neuron_indices.len()
            || !neuron_indices.windows(2).all(|pair| pair[0] < pair[1])
        {
            return false;
        }
        self.a_at_relu.remove(&node_name);
        self.beta_sparse_a_at_relu.insert(
            node_name,
            GraphBetaSparseA {
                neuron_indices,
                values,
            },
        );
        true
    }

    /// Resolve the lower-A column used by beta-gradient consumers.
    ///
    /// Dense storage takes precedence and preserves every existing graph-alpha
    /// path. Compact storage is visible only through this beta-specific accessor.
    pub(crate) fn beta_a_column(
        &self,
        node_name: &str,
        neuron_idx: usize,
    ) -> Option<ArrayView1<'_, f32>> {
        if let Some(dense) = self.a_at_relu.get(node_name) {
            return (neuron_idx < dense.ncols()).then(|| dense.column(neuron_idx));
        }
        let sparse = self.beta_sparse_a_at_relu.get(node_name)?;
        let compact_col = sparse.neuron_indices.binary_search(&neuron_idx).ok()?;
        Some(sparse.values.column(compact_col))
    }

    #[cfg(test)]
    pub(crate) fn has_beta_sparse_a(&self, node_name: &str) -> bool {
        self.beta_sparse_a_at_relu.contains_key(node_name)
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

/// `#cgan-child-zono` (default-dark): certified per-child DeepZ zonotope
/// intermediates for the input-split deferred rebound
/// (`bab.alpha_crown.cgan_child_zono_refresh`).
///
/// When `enabled`, every deferred child of the input-split BaB gets a
/// certified binary64 zonotope pass over ITS OWN box (the shared
/// `crate::zono_core` engine), each ReLU's pre-activation box intersected with
/// the inherited map (root CROWN map, then the domain's own override) before
/// DeepZ picks its slope. The resulting pre-activation boxes are merged
/// shrink-only into the map the batched CROWN rebound and the fresh
/// exact-domain clip consume. `enabled: false` (the default) is byte-identical
/// to the historical lane: the prepass does no work and prints nothing.
///
/// SOUNDNESS. The child box is a sub-box of the root box, so the root map and
/// the child's own zonotope both enclose the child's reachable set, and so
/// does their intersection; DeepZ's identity holds for any valid enclosure.
/// Every rounding and the BatchNorm coefficient error is charged; any
/// refusal, NaN or deadline leaves the domain on its previous map.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CganChildZonoRefreshConfig {
    /// Arm the per-child zonotope refresh.
    #[serde(default)]
    pub enabled: bool,
    /// Generator budget per child; rows beyond it fold (soundly) into the
    /// remainder. Default 16384: Z1 measured a 1024 cap costing 83 leaves
    /// instead of 15 on row 7 and never saw more than 4,153 generators, so the
    /// default must not bind. `0` is refused, never read as "unlimited".
    #[serde(default = "CganChildZonoRefreshConfig::default_max_generators")]
    pub max_generators: u32,
    /// Per-child wall budget in seconds. `None`: bounded only by the BaB
    /// deadline. Expiry refuses that child (it keeps its previous map).
    #[serde(default)]
    pub max_secs: Option<f32>,
}

impl CganChildZonoRefreshConfig {
    /// The default generator budget (Z1 §3.3 item 3).
    pub const DEFAULT_MAX_GENERATORS: u32 = 16_384;
    /// Smallest admissible generator budget.
    pub const MIN_MAX_GENERATORS: u32 = 8;
    /// Largest admissible generator budget.
    pub const MAX_MAX_GENERATORS: u32 = 1 << 20;

    const fn default_max_generators() -> u32 {
        Self::DEFAULT_MAX_GENERATORS
    }

    /// Whether a configured generator budget is admissible.
    #[must_use]
    pub fn max_generators_is_valid(value: u32) -> bool {
        (Self::MIN_MAX_GENERATORS..=Self::MAX_MAX_GENERATORS).contains(&value)
    }

    /// Whether a configured per-child budget is admissible (finite, positive).
    #[must_use]
    pub fn max_secs_is_valid(value: f32) -> bool {
        value.is_finite() && value > 0.0
    }
}

impl Default for CganChildZonoRefreshConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_generators: Self::DEFAULT_MAX_GENERATORS,
            max_secs: None,
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

/// abc's `minimum_sparsity` default (`backward_bound.py:580-589`), which is
/// also the fraction the `NY_CROWN_IBP_SPARSE_RELU_ROWS` route has always
/// applied. Keeping it as the serde default makes an unset key byte-identical.
const fn default_cgan_unstable_rows_max_frac() -> f32 {
    0.9
}

/// Serde default for `early_stop_patience` field (backward compat with configs
/// serialized before the field was added).
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

/// Serde/default value for the aggregate DAG reference-refresh budget.
///
/// This preserves the historical policy: the complete sequence of refreshes
/// may consume at most 25% of the root alpha loop's remaining deadline.
fn default_reference_refresh_fraction() -> f32 {
    AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION
}

/// Configuration for alpha-CROWN optimization.
///
/// This pre-1.0 configuration surface grows as proof scheduling gains typed
/// resource and fallback policies. Downstream Rust callers should construct it
/// from [`Self::default`] and override selected fields instead of using
/// exhaustive struct literals. Newly added serialized fields carry explicit
/// serde defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlphaCrownConfig {
    /// Number of optimization iterations.
    pub iterations: usize,
    /// #spec-axis-alpha (design §4): number of per-spec δ slots. The K worst
    /// margins at loop entry get private α corrections
    /// (`α_eff = clamp01(α_base + δ_slot)`); 0 (default) = shared-α behavior,
    /// byte-identical. Slots update only when the margin-gradient lane binds
    /// their row, so a nonzero K without that lane is inert by construction.
    pub alpha_spec_slots: usize,
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
    /// Root-only cGAN CROWN-IBP collection policy.
    ///
    /// When enabled together with `fix_interm_bounds = false`, a sequential
    /// ConvTranspose graph keeps its certified forward-linear reference map as
    /// the baseline and spends the root collection budget on one demanded
    /// ReLU-preactivation target. Only unresolved ReLU rows are sent through
    /// CROWN when they satisfy the established 90% sparse threshold; otherwise
    /// that same target uses the dense/chunked collector. A target is published
    /// only after its selected row set completes, then its enclosure is
    /// intersected with the baseline and propagated through downstream nodes.
    ///
    /// This policy is deliberately category-shaped and default-dark. Child
    /// input-split warm starts clear it explicitly and retain
    /// `fix_interm_bounds = true`, so they continue to use the cheap
    /// forward-linear route.
    #[serde(default)]
    pub cgan_sparse_target_complete_root: bool,
    /// Root-only cGAN complete CROWN-IBP cascade.
    ///
    /// The collector starts from the certified forward-linear map, tightens
    /// every demanded pre-activation target in topological order, and reuses
    /// the resulting shrink-only map for the root objective. Child input-split
    /// warm starts clear this flag explicitly, so their inexpensive
    /// forward-linear route is unchanged.
    ///
    /// This is distinct from [`Self::cgan_sparse_target_complete_root`]: the
    /// sparse policy deliberately buys one target, while this policy pays for
    /// the complete cascade needed by the cGAN root certificate.
    #[serde(default)]
    pub cgan_complete_crown_ibp_root: bool,
    /// Publish the complete cGAN root cascade map to the input-split BaB under
    /// `bound_prop_method: crown` (`#crown-root-map`).
    ///
    /// [`Self::cgan_complete_crown_ibp_root`] names the collector; this key
    /// names a CONSUMER. The input-split root collector selects its arm by
    /// `bound_prop_method` alone, and its `crown` arm historically returned no
    /// intermediate map at all, so on the shipped cgan lane (`crown`) the
    /// cascade that every cgan yaml arms was consulted by nothing: the BaB
    /// children bounded on IBP/forward-linear intermediates (measured
    /// 2026-09-10, `root_node_bounds=None`, root margin -5.8945 against abc's
    /// plain-CROWN -0.0034 on the same row). When set, the `crown` arm runs the
    /// same reference collector the alpha arm runs and publishes its map (no
    /// alpha state) — but only when BOTH publication guards hold
    /// (`input_split_ibp_enhancement` and `input_split_stacked_rebound`, see
    /// `#root-map-two-guards`), only when the graph is typed-cGAN eligible,
    /// and only when the collector reports the typed cascade as its actual
    /// source. Anything else declines to the historical `(None, None)`.
    ///
    /// SOUNDNESS. The published map is the certified forward-linear baseline
    /// intersected shrink-only with the CROWN-IBP cascade; under the two
    /// guards every consumer MERGES it with its own per-domain bounds, so it
    /// can only tighten. Default-dark until a sealed official-budget row
    /// measures it.
    #[serde(default)]
    pub cgan_crown_root_map: bool,
    /// Run the typed cGAN CROWN-IBP root cascade in DENSE (matrix) conv mode
    /// (`#cgan-collector-dense`, experiment E5; default-dark).
    ///
    /// The collector historically forces Patches for spatial conv targets on
    /// BOTH of its routes: the graph-native loop ORs
    /// `collector_patches_override` into the `use_patches_mode` gate
    /// (`target_backward_patches.rs`), and the sequential linear-chain
    /// shortcut (`network/ibp/crown_ibp.rs`) has no conv-mode input at all.
    /// So `conv_mode: matrix` never reached intermediate-bound collection
    /// (measured 2026-09-10 on cgan_2023 nCh_3 prop_0: the cascade's own
    /// CROWN was looser than the forward-linear baseline on ~125k of ~155k
    /// elements while the generator's unstable counts matched abc exactly).
    /// When set, the cascade (a) declines the sequential shortcut, (b) passes
    /// `collector_patches_override=false` to every target walk, and (c) does
    /// not let the forward-linear MAC proxy refuse a walk upfront, so on a
    /// `conv_mode: matrix` graph every conv backward is the dense A-matrix
    /// product alpha-beta-CROWN runs. Cost, not soundness: dense and patches
    /// are two evaluations of the same linear relaxation. Observable:
    /// `[collector-mode] cgan_collector_dense=… patches_override=…
    /// use_patches_mode=…`, once per typed cascade, armed or not.
    #[serde(default)]
    pub cgan_collector_dense: bool,
    /// `#cgan-unstable-rows` (default-dark): inside the typed, shrink-only
    /// cGAN root cascade, seed each demanded ReLU-preactivation target's CROWN
    /// backward with ONLY the rows that the certified forward-linear baseline
    /// leaves sign-unstable (`l < 0 < u`) or unresolved, instead of the full
    /// `[dim x dim]` identity.
    ///
    /// This is alpha-beta-CROWN's `sparse_intermediate_bounds`
    /// (`auto_LiRPA/backward_bound.py:568-604` builds the one-hot C over
    /// `get_unstable_locations`' index set; `bound_general.py:1097-1126`
    /// supplies the reference). The mechanism already exists in this collector
    /// — it was reachable only through the raw `NY_CROWN_IBP_SPARSE_RELU_ROWS`
    /// environment variable, which `vnncomp_scripts/run_instance.sh` never
    /// exports, so it could not fire in competition. This key is the typed
    /// delivery of the same bool; the environment variable still wins.
    ///
    /// SOUNDNESS: cost, not authority. Omitted rows keep the certified
    /// forward-linear enclosure the cascade was seeded with, selected rows are
    /// `baseline ∩ CROWN_row` from the same `run_target_backward_pass` the
    /// full-width walk uses, and any disjoint element rejects the whole target
    /// and retains the baseline. A wrong row mask therefore yields a LOOSER
    /// map, never an unsound one. Eligibility is restricted to targets whose
    /// every bound-demanding consumer is a ReLU (`demand.rs`), whose
    /// relaxation is magnitude-independent for a sign-stable row — abc's own
    /// admissibility condition (`bound_general.py:1102-1104`).
    ///
    /// Masked by `CrownIbpCollectionMode::requires_shrink_only_publication()`,
    /// so an ordinary (Standard) collection is byte-identical with the key set.
    /// Observable, armed or not: `[unstable-rows] root armed=… source=…`.
    #[serde(default)]
    pub cgan_unstable_rows: bool,
    /// Maximum fraction of a target's rows that may be selected before
    /// `cgan_unstable_rows` gives up on sparsity for that target and takes the
    /// existing full-width/chunked walk.
    ///
    /// Default `0.9` — alpha-beta-CROWN's `minimum_sparsity`
    /// (`backward_bound.py:580-589`), and the value the environment route has
    /// always used. Stored to the collection policy as integer per mille, so
    /// `0.9` reproduces the historical `selected * 10 <= total * 9` predicate
    /// exactly and cache identity never compares floats. Clamped to `[0, 1]`.
    ///
    /// SCHEDULING ONLY: it decides whether a target is walked sparsely or
    /// densely, never what is written to a row.
    #[serde(default = "default_cgan_unstable_rows_max_frac")]
    pub cgan_unstable_rows_max_frac: f32,
    /// Row ceiling for demanded CROWN targets inside the complete cGAN root
    /// cascade (`#cgan-narrow-target-budget`).
    ///
    /// `None` (default) keeps the historical all-demanded candidate set. When
    /// set, a demanded pre-activation target whose objective-row count exceeds
    /// this ceiling is DECLINED as a tightening candidate and retains the
    /// certified forward-linear baseline the cascade was seeded with.
    ///
    /// This is a SCHEDULING policy, not a bound algorithm. Declining a target
    /// can only leave it at an enclosure the collector already holds, so the
    /// published map stays shrink-only with respect to that baseline and proof
    /// authority is unchanged in either direction.
    ///
    /// Motivation: on the official cGAN rows the collection budget is split in
    /// proportion to objective-row count, so the wide generator targets (up to
    /// 28,800 and 61,504 rows) draw ~86% of it. Those targets are already
    /// within 1.02-1.13x of the sampled-exact width under the forward-linear
    /// seed, while the narrow discriminator targets they starve are where the
    /// root over-approximation is actually manufactured.
    #[serde(default)]
    pub cgan_narrow_target_max_rows: Option<usize>,
    /// Per-child wall budget, in seconds, for CHILD-SIDE intermediate
    /// refinement (`#cgan-child-interm-refine`).
    ///
    /// `None` (default) is today's behaviour exactly: the complete cGAN
    /// cascade is root-only, and every BaB child is bounded against the frozen
    /// ROOT intermediate map plus the objective backward pass.
    ///
    /// When set — and only when `cgan_complete_crown_ibp_root` is also armed,
    /// so a child can never enter a lane its root did not — each deferred child
    /// re-runs the cascade on ITS OWN box under this wall, and the resulting
    /// map is intersected into that child's intermediate bounds.
    ///
    /// The budget is deliberately the ONLY dial, and it is deliberately not
    /// optional: an unbounded per-child cascade is the O(N^2) pass that
    /// `fix_interm_bounds = true` exists to avoid, so arming the lane without
    /// saying what it may spend is not representable.
    ///
    /// SOUNDNESS. This selects a different sound enclosure, never a looser
    /// relaxation. The cascade is the same collector the root lane ships, run
    /// on the child's box: a certified forward-linear seed plus a shrink-only
    /// CROWN pass over it. Spending less time can only make the collector
    /// salvage back to that seed — the bound the dark path would have
    /// published anyway — so the budget is verdict-neutral in both directions.
    #[serde(default)]
    pub cgan_child_interm_refine_max_secs: Option<f64>,
    /// `#cgan-child-seed` (default-dark): seed each child cascade's certified
    /// forward-linear baseline with the domain's INHERITED intermediate map —
    /// the frozen root CROWN map intersected with any child-local override
    /// the domain already carries — before the shrink-only CROWN cascade runs.
    ///
    /// Only meaningful while `cgan_child_interm_refine_max_secs` is armed (the
    /// prepass it modifies does not exist otherwise). `false` is today's
    /// behaviour exactly: the child cascade starts from the forward-linear map
    /// on its own box, so the sparse ReLU-row selector walks every row the
    /// (roughly ten times looser) forward-linear seed leaves unstable in the
    /// discriminator, rather than the rows the root CROWN map leaves unstable.
    ///
    /// SOUNDNESS. Intersection of certified enclosures of the SAME reachable
    /// set: the root map encloses the root box's image, and the root box
    /// contains the child box; a child-local override is, by its own contract,
    /// an enclosure of the child box's image; the forward-linear baseline is
    /// computed on the child box itself. The intersection therefore still
    /// contains the child's reachable set, and the cascade only ever shrinks
    /// its baseline. Any node whose seed entry is shape-mismatched, NaN, or
    /// disjoint from the baseline in any element keeps its baseline entry
    /// untouched (fail closed), and no node absent from the baseline is added.
    #[serde(default)]
    pub cgan_child_interm_refine_seed_from_parent: bool,
    /// `#cgan-child-seed` (default-dark): run up to this many child cascades
    /// of one deferred batch CONCURRENTLY. `0` and `1` are the serial
    /// reference, i.e. today's behaviour.
    ///
    /// Scheduling only: every child's map is a pure function of its own box,
    /// the shared config and the domain's inherited map, and results are
    /// applied in the original domain order, so concurrency cannot change what
    /// is published (bit-identical to serial whenever no deadline or memory
    /// budget binds). The requested width is only a CEILING: each tranche is
    /// re-admitted against a per-worker RSS estimate, the live process memory
    /// headroom (which on Linux includes `MemAvailable` even with no cgroup,
    /// i.e. the competition configuration), and an absolute byte cap; an
    /// unobservable headroom runs serially.
    #[serde(default)]
    pub cgan_child_interm_refine_parallel: usize,
    /// `#cgan-child-seed` (default-dark): hand a popped domain's own refined
    /// intermediate map to the fresh exact-domain clip (`process_batch.rs`),
    /// which otherwise recomputes its planes against the frozen root map only.
    ///
    /// Only meaningful while `cgan_child_interm_refine_max_secs` is armed: the
    /// deferred rebound then keeps the child-local map it just consumed on the
    /// domain until the clip takes it. The map is an enclosure of that domain's
    /// box (see `cgan_child_interm_refine_seed_from_parent`), and the clip's
    /// bounder intersects it with the root map and fresh IBP exactly as the
    /// rebound did, so the planes it builds are valid on the domain box.
    #[serde(default)]
    pub cgan_child_interm_refine_fresh_clip_map: bool,
    /// `#cgan-child-zono` (default-dark): certified per-child zonotope
    /// intermediates for the input-split deferred rebound and the fresh clip.
    /// See [`CganChildZonoRefreshConfig`].
    #[serde(default)]
    pub cgan_child_zono_refresh: CganChildZonoRefreshConfig,
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
    /// causing CROWN bounds identical to IBP.
    pub adaptive_skip: bool,
    /// Depth threshold for adaptive skipping.
    ///
    /// Networks with more than this many ReLU layers will skip α-CROWN optimization
    /// if `adaptive_skip` is enabled. This threshold is retained for explicit opt-in
    /// uses of the heuristic even though the default configuration disables it.
    ///
    /// Default: 20 (raised from 8 for medium-depth DAGs)
    pub adaptive_skip_depth_threshold: usize,
    /// Run a pilot iteration to check if α-CROWN helps before full optimization.
    ///
    /// When enabled with `adaptive_skip`, runs 1 iteration and compares the improvement
    /// to CROWN bounds. If improvement is below `pilot_improvement_threshold`, skips
    /// remaining iterations.
    ///
    /// This catches cases where depth isn't the only factor (e.g., already tight bounds).
    /// Default: false (pilot exits too early for deep DAG networks)
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
    /// When enabled, the candidate violation constraints (`A·y <= rhs`) are
    /// propagated backward to tighten the region that remains for BaB.
    ///
    /// Default: disabled (InvpropConfig::default())
    #[serde(default)]
    pub invprop: InvpropConfig,

    /// Output constraints for INVPROP optimization.
    ///
    /// [`OutputConstraints`] is a generic linear-region representation, but
    /// this verifier-facing field has a strict polarity contract: its
    /// conjunctive inequalities describe the candidate **violation** region.
    /// When `invprop.enabled` is true, they initialize nonnegative duals that
    /// condition backward propagation on that region and are optimized
    /// alongside alphas. Certifying the conditioned region infeasible proves
    /// that the original property holds; supplying the property-holding region
    /// here would reverse that reasoning.
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
    /// to false — without it, conv alpha count is ~63x too high.
    ///
    /// Default: true (preserves current per-neuron behavior until channel-shared
    /// layout is activated in later slices).
    /// Source: alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/arguments.py:348.
    #[serde(default = "default_full_conv_alpha")]
    pub full_conv_alpha: bool,

    /// Fraction of the root alpha loop's remaining deadline assigned to the
    /// complete sequence of intermediate reference-bound refreshes.
    ///
    /// This is one aggregate pool, not a fresh fraction per iteration. Valid
    /// values are in `[0.01, 1.0]`. The default `0.25` preserves the historical
    /// scheduling policy for every preset that does not opt in.
    #[serde(default = "default_reference_refresh_fraction")]
    pub reference_refresh_fraction: f32,

    /// #joint-interm-alpha: recompute the INTERMEDIATE bounds with the CURRENT
    /// alpha every `k` ascent iterations, making the ascent a block-coordinate
    /// (Gauss-Seidel) optimization over BOTH alpha and the relaxation instead of
    /// an optimization of alpha over a relaxation that is frozen forever.
    ///
    /// `0` (default) = legacy: the historical `improved_output`-gated refresh.
    /// `k >= 1` = joint mode, refresh on every `k`-th iteration.
    ///
    /// # Why this is the algorithm, not a schedule knob
    ///
    /// ny builds its intermediate map ONCE, before the ascent, from a
    /// forward-linear reference computed with **no alpha at all**, then sets
    /// `fix_interm_bounds` and reads that frozen map on every iteration. So the
    /// ascent solves
    ///
    /// ```text
    ///     max_alpha  f(alpha ; I_0)          I_0 fixed, alpha-independent
    /// ```
    ///
    /// alpha-beta-CROWN instead deletes every cached intermediate bound each
    /// iteration and recomputes it under autograd with the current alpha, so it
    /// solves `max_alpha f(alpha ; I(alpha))`. Reviewing its source against ours:
    /// *"a reimplementation that computes intermediate bounds once and then only
    /// optimizes the last-layer slopes is solving a strictly weaker problem, and
    /// no amount of extra iterations or throughput on that problem will close the
    /// gap."* That is the measured situation — 20 abc iterations reach ~91/99 at
    /// the root where ny reaches 0/99, and ny's own uncapped 3242s ascent moved
    /// the scored frontier by exactly zero.
    ///
    /// ny has no autograd, so the gradient route through `I(alpha)` is not
    /// available here (that needs a second adjoint harvest — see the ladder in
    /// the design notes). What IS available, and is what this key turns on, is
    /// the alternating form: ascend alpha, rebuild `I` at the new alpha, ascend
    /// again on the tighter relaxation. Block-coordinate ascent on the same
    /// objective — strictly stronger than holding `I` at `I_0`, and it is also
    /// the mechanism behind abc's own `best_intermediate_bounds` carry-forward
    /// (`optimized_bounds.py:338-367,500-615`), which this codebase already
    /// ported in `reference_bounds.rs` and then left disarmed.
    ///
    /// # Soundness
    ///
    /// Neutral, on three legs: every alpha is clamped to `[0,1]` and any such
    /// alpha is a certified-sound relaxation (`alpha_sound_regardless`, machine
    /// checked); the commit path is `merge_tighter_bounds`, an element-wise
    /// max/min that can only shrink the box; and every refusal (deadline,
    /// capacity, shape) keeps the previous sound reference. Refreshing more often
    /// therefore trades schedule for tightness and can never trade soundness.
    #[serde(default)]
    pub joint_interm_alpha_every: usize,

    /// #envelope-grad: use the concretization-argmin alpha-gradient rule instead
    /// of the shipped local rule.
    ///
    /// The shipped rule is `SUM_j [A[j,i]]_+ * l_i`. Every factor is
    /// sign-definite — `l_i < 0` by the unstable guard, and only `A > 0` terms
    /// are admitted — so it is `<= 0` for every neuron, objective and iteration.
    /// Adam can then only DECREASE alpha, monotonically to the 0 clamp; it
    /// carries no direction, which is why the measured lr sweep 0.25 / 0.05 /
    /// 0.01 leaves the bound BIT-IDENTICAL. Machine-checked in
    /// `crates/ny-cert/proofs/lean/NyProof/AlphaGradientDefect.lean`.
    ///
    /// The envelope rule swaps the factor for `clamp(h_hat(x*), l, u)`, the
    /// relaxed pre-activation at the binding row's concretization argmin. That
    /// value CAN be positive, so the field is not sign-definite and alpha can
    /// rise. Measured 2026-08-17 on cifar100 at the official budget: the census
    /// MOVES for the first time in this campaign (idx_8600 83/99 -> 85/99 on a
    /// row that is 83/99 in every other arm; idx_6659 67/99 -> 78/99), against
    /// every prior alpha lever leaving it bit-identical. Mixed sign across rows
    /// and NO verdict conversion is claimed.
    ///
    /// Soundness-neutral: gradients only select which alpha the ascent lands on,
    /// and every alpha in `[0,1]` is a certified-sound relaxation
    /// (`alpha_sound_regardless`). A wrong gradient costs tightness, never
    /// validity. The real risk is SCHEDULE — the rule adds a point-forward pass
    /// per iteration — which is why this is a per-benchmark key rather than a
    /// compiled default.
    ///
    /// READERS (#alpha-envelope-grad-cpu): the GPU warmup/rescale sites AND the
    /// CPU chain-rule gradient on both the DAG lane (AnalyticChain replay) and
    /// the sequential lane. The CPU sites used to resolve env-only, so this
    /// field was dark on every graph ascent that does not take the GPU warmup
    /// lane; arrival is now printed as `[alpha-envelope-grad] cpu=1 ...
    /// source=config`.
    #[serde(default)]
    pub alpha_envelope_grad: bool,

    /// Optional absolute ceiling, in seconds, on the same aggregate reference-
    /// refresh pool. The effective pool is
    /// `min(remaining * reference_refresh_fraction, max_secs)`.
    ///
    /// `None` preserves the fraction-only default. `Some(0)` deliberately
    /// disables refresh work while retaining the previous sound reference map.
    /// This affects scheduling only; every completed candidate remains the same
    /// certified enclosure.
    #[serde(default)]
    pub reference_refresh_max_secs: Option<u64>,

    /// On a deadline refusal from the preferred forward-linear intermediate
    /// collector, return a plain IBP reference map instead of entering the
    /// historical CROWN-IBP fallback.
    ///
    /// This is a scheduling-only endgame policy for small-input image graphs:
    /// both maps are certified enclosures, while IBP is generally looser and
    /// much cheaper. It is default-off so callers and presets that do not opt
    /// in preserve the historical fallback and bound quality exactly.
    #[serde(default)]
    pub forward_linear_deadline_fallback_to_ibp: bool,

    /// Root-bootstrap scheduling hint: when the caller will retain only the
    /// initialized alpha state and the fixed intermediate reference map, a
    /// zero-update DAG collection may skip its otherwise-obligatory initial
    /// output CROWN evaluation.
    ///
    /// This is deliberately runtime-only and defaults off. Direct bounds-only
    /// callers preserve the historical `iterations == 0` contract: evaluate
    /// the initial CROWN slopes once and return that bound.
    #[serde(skip)]
    pub skip_zero_iteration_collection_initial_bound: bool,

    /// Wall-clock deadline for alpha optimization.
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

    /// Optional multi-row spec objective for RANKING α iterates in the root warmup
    /// (#root-alpha-margin). When `Some` and the effective gate is armed by the
    /// typed preset default or `NY_ROOT_ALPHA_MARGIN`, the DAG warmup scores each
    /// iterate with [`AlphaSpecAscent::hinge_score`] and returns the best-scoring
    /// α instead of the last one.
    ///
    /// SELECTION ONLY — see [`AlphaSpecAscent`]. `None` (every caller today) ⇒ the
    /// loop's objective, early stop, and returned α are byte-identical to the
    /// legacy raw-sum path.
    #[serde(skip)]
    pub spec_ascent: Option<AlphaSpecAscent>,

    /// Preset-supplied default for #root-alpha-margin (rank the root warmup's α iterates by
    /// the spec objective and keep the best, rather than the last iterate).
    ///
    /// `NY_ROOT_ALPHA_MARGIN` still overrides this in both directions. Default `false`, so a
    /// config that never names it is byte-identical.
    #[serde(default)]
    pub root_alpha_margin: bool,

    /// Preset-supplied default for #alpha-zero-yield: retire the root α ascent after this
    /// fraction of its own window passes with no improvement over the best iterate, returning
    /// the remaining window to search. Sound by construction — the early exit returns the
    /// already-certified elementwise best. Stopping sooner can return a looser certified
    /// enclosure; it cannot manufacture an invalid bound.
    ///
    /// Valid range `(0.0, 0.9)`; see [`AlphaCrownConfig::alpha_zero_yield_frac_is_valid`].
    /// `NY_ALPHA_ZERO_YIELD_FRAC` still overrides this wherever it is PRESENT, including as a
    /// kill switch (any invalid value, e.g. `0`, disarms a preset-armed fraction). Default
    /// `None`, so a config that never names it is byte-identical.
    #[serde(default)]
    pub alpha_zero_yield_frac: Option<f64>,

    /// #inert-ascent-exit (typed key `bab.alpha_crown.inert_ascent_exit_iters`): retire the DAG
    /// α ascent after this many CONSECUTIVE iterations whose best improvement was EXACTLY zero
    /// — the iterate never beat the best, so the frame is not moving (an IBP-grade root, an
    /// α family with nothing to move). The historical `early_stop_patience` rule (10
    /// iterations below `tolerance`) keeps running alongside; whichever fires first wins.
    ///
    /// Sound by construction: the exit takes the same `should_save_best` route as ordinary
    /// convergence and returns the already-certified elementwise best. Stopping sooner can
    /// return a looser certified enclosure; it cannot manufacture an invalid bound.
    ///
    /// Valid values are `>= 1`; `Some(0)` is treated as unset. Default `None`, so a config
    /// that never names it is byte-identical (`propagate_dag::ascent_patience_exit`).
    #[serde(default)]
    pub inert_ascent_exit_iters: Option<usize>,

    /// #vit-crown-interm (typed key `solver.alpha-crown.crown_ibp_reference_cap_secs`):
    /// wall cap, in whole seconds, on the `fix_interm_bounds = false` reference route.
    ///
    /// The typed `fix_interm_bounds: false` route collects a per-node CROWN-IBP
    /// reference map before the DAG alpha loop (`bounds/alpha.rs`), may collect it a
    /// second time after the loop, and then runs a full all-node alpha pass. On
    /// vit_2023 `ibp_3_3_8` one uncapped collection cost 46-50 s of the 100 s budget
    /// and turned a row plain IBP decides in ~10 s (`ibp_3_3_8_7322`) into a timeout
    /// (`docs/VIT_INTERM_FLOOR_CONVERSION_2026-08-08.md` §5). Setting this key bounds
    /// the route's cost in three places, and ONLY on that route
    /// (`fix_interm_bounds = false`; a config with `true` — which every input-split
    /// warm-start child is forced to — resolves no cap and is inert):
    ///
    /// 1. the reference collection's CROWN tightening sweep gets this cap
    ///    (`CrownIbpCollectOptions::tightening_cap`): targets it does not finish KEEP
    ///    their certified IBP box, finished ones keep their CROWN∩IBP box, and the map
    ///    is always complete — the cap can never error the root out;
    /// 2. the DAG loop's pre-loop initial bound and the post-loop publication reuse
    ///    that map (the optimizer's shrink-only artifact) instead of re-collecting it
    ///    under a possibly expired alpha deadline, which would publish plain IBP;
    /// 3. the post-loop all-node alpha pass stops after this many seconds and fills
    ///    every node it did not reach from that same reference map.
    ///
    /// SOUND by construction: every published node box is either a completed CROWN
    /// enclosure intersected with its certified IBP box, or that IBP box. The cap only
    /// decides how many targets get the tighter of the two.
    ///
    /// A present `NY_CROWN_IBP_COLLECTOR_CAP_SECS` still wins for step 1 (the existing
    /// research override). Valid values are `1..=3600`; the preset loader rejects
    /// anything else and [`Self::crown_ibp_reference_cap`] ignores it fail-closed.
    /// Default `None`, so a config that never names it is byte-identical.
    #[serde(default)]
    pub crown_ibp_reference_cap_secs: Option<u64>,

    /// #vit-margin-alpha (B1–B3): typed form of the former env-only
    /// `NY_ROOT_ALPHA_MARGIN_GRADIENT` child gate (lever
    /// `ny_levers::decls::root_alpha::ROOT_ALPHA_MARGIN_GRADIENT`, preset key
    /// `bab.alpha_crown.root_alpha_margin_gradient`). Subordinate to
    /// [`Self::root_alpha_margin`]: without the parent gate it is never read.
    ///
    /// Armed, the root α loss becomes the spec's margin C-rows. On a graph the
    /// resident joint-adjoint lane already accepts (ReLU-only optimizer state)
    /// that lane keeps its historical behavior; on a graph carrying bilinear
    /// α state (vit: every var×var MatMul is lowered to `BilinearCrown`), the
    /// certified C-row margin ascent (`propagate_dag/margin_alpha.rs`) runs
    /// instead and publishes a certify-once margin bound for the root.
    ///
    /// SOUND: steering and selection only. Every α ∈ [0,1] and every
    /// interpolation parameter r ∈ [0,1] selects a certified relaxation, and
    /// the only bound published is a fresh certified pass. Default `false`, so
    /// a config that never names it is byte-identical. The env var still wins
    /// wherever it is present (`ny_levers::read_over_config` layering).
    #[serde(default)]
    pub root_alpha_margin_gradient: bool,

    /// #vit-margin-alpha (B4): optimize the bilinear McCormick interpolation
    /// parameters (4 per product element, `[4, m, n, k]`, shared over batch
    /// dims) of every `BilinearCrown` and var×var `MatMul` node inside the
    /// certified C-row margin ascent, checkpointed together with the ReLU α
    /// and published through the same certify-once pass. Lever
    /// `ny_levers::decls::root_alpha::ROOT_ALPHA_MATMUL`, preset key
    /// `bab.alpha_crown.matmul_alpha`. Inert unless that lane runs.
    ///
    /// SOUND: every r ∈ [0,1] yields a convex combination of two valid
    /// McCormick planes whose constant is certified outward
    /// (`bilinear::mccormick::certify_plane_constant`). Default `false`
    /// (fixed midpoint planes, byte-identical).
    #[serde(default)]
    pub matmul_alpha: bool,

    /// #vit-margin-alpha-complex (P6): let the certified C-row margin ascent
    /// (`propagate_dag/margin_alpha.rs`) take a root whose graph ALSO carries
    /// MulBinary and/or Reciprocal α state, with those two families' α FROZEN
    /// at their DAG-init values for the whole lane (every fold, the committed
    /// checkpoint and the certify-once pass). The fixed-index softmax
    /// decomposition (`softmax: complex_fixed_index`) plants exactly one
    /// Reciprocal and one MulBinary per softmax, which is why the lane refused
    /// every vit root that used it (`[alpha-margin] armed=0
    /// reason=unsupported_alpha_family`). Only the ReLU α (+ per-row δ) and the
    /// attention McCormick parameters (`matmul_alpha`) move. Preset key
    /// `bab.alpha_crown.margin_alpha_freeze_mul_reciprocal` (or the
    /// `solver.alpha-crown` twin). Inert unless the margin lane is requested.
    ///
    /// SOUND: any FIXED relaxation is valid — the frozen MulBinary planes are
    /// the certified interpolated McCormick planes at r = 0.5 (the DAG init),
    /// the frozen Reciprocal lines are its init tangent/chord — and the
    /// published bound is one fresh certified pass with exactly that state
    /// (abc's `disable_optimization` analogue). Sigmoid/Tanh/Sqrt α still
    /// refuse. Default `false`: the lane refuses those graphs exactly as before.
    #[serde(default)]
    pub margin_alpha_freeze_mul_reciprocal: bool,

    /// #vit-root-alpha (ENGINE_PROGRAM §2 V2): which SEARCH runs inside the
    /// certified C-row margin lane (`propagate_dag/margin_alpha.rs`).
    ///
    /// * `ascent` (the default): today's certified ascent, bit for bit — every
    ///   iterate a certified fold, then the certify-once pass.
    /// * `joint_f32`: the lane lowers its graph into a plain-data search copy
    ///   (self-checked against its own point forward), hands it with the
    ///   certified boxes FROZEN to the injected untrusted backend (ny-bbe's CPU
    ///   f32 search: ReLU slopes, the attention products' McCormick planes and
    ///   the fixed-index softmax's MulBinary planes, jointly, against the margin
    ///   rows), ingests the returned parameters through a clamp gate (non-finite
    ///   → incumbent, the rest into `[0, 1]`), runs ONE certified fold with
    ///   them, keeps that state only when its margin score is STRICTLY better
    ///   than the incumbent's, and publishes the per-row intersection of the two
    ///   certified folds (never looser than the incumbent; WITHDRAWN, with an
    ///   unconditional `[bbe-alarm]` line, if the two are ever disjoint).
    ///
    /// Preset key `bab.alpha_crown.margin_alpha_search` (or the
    /// `solver.alpha-crown` twin). Inert unless the margin lane runs. The
    /// certified fold runs through the lane's own backward, so it takes
    /// `bab.bilinear_gemm_kernel` (V1) when that key is armed.
    ///
    /// SOUND: the search reaches the bound only as parameters the gate clamps
    /// into their valid domain; every published number is a certified fold.
    #[serde(default)]
    pub margin_alpha_search: MarginAlphaSearch,
}

/// Which search the certified C-row margin lane runs
/// (`AlphaCrownConfig::margin_alpha_search`, #vit-root-alpha).
///
/// The serde tokens are the preset key's values ([`Self::TOKENS`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MarginAlphaSearch {
    /// The certified ascent (default; bit-identical to the lane before V2).
    #[default]
    #[serde(rename = "ascent")]
    Ascent,
    /// The injected untrusted joint f32 search (V2), certified once.
    #[serde(rename = "joint_f32")]
    JointF32,
}

impl MarginAlphaSearch {
    /// The preset vocabulary, in declaration order.
    pub const TOKENS: [&'static str; 2] = ["ascent", "joint_f32"];

    /// The canonical token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Ascent => "ascent",
            Self::JointF32 => "joint_f32",
        }
    }
}

impl Default for AlphaCrownConfig {
    fn default() -> Self {
        Self {
            // α,β-CROWN default: 100 iterations for incomplete verifier.
            // Early stop patience (10) terminates early when bounds converge.
            // Source: alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/arguments.py:354 (init_iteration=100).
            iterations: 100,
            // Spec-axis δ is opt-in per category (#spec-axis-alpha).
            alpha_spec_slots: 0,
            // α,β-CROWN default: lr_init_alpha=0.1.
            // Source: alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/arguments.py:351.
            learning_rate: 0.1,
            lr_decay: 0.98,
            tolerance: 1e-4,
            use_momentum: true,
            momentum: 0.9,
            // AnalyticChain: true chain-rule gradients, closest to reference α,β-CROWN's
            // loss.backward() (optimized_bounds.py:870). SPSA with 1 sample is noise-dominated
            // for networks with many unstable neurons.
            gradient_method: GradientMethod::AnalyticChain,
            spsa_samples: 1, // Only used when gradient_method is Spsa
            // Default to true: use IBP bounds for intermediates (fast O(N)).
            // Matches α,β-CROWN's fix_interm_bounds=True default.
            fix_interm_bounds: true,
            // cGAN single-target root collection is a measurement-only policy
            // until a sealed official-budget row converts.
            cgan_sparse_target_complete_root: false,
            // Complete cGAN root collection is separately default-dark until
            // its production-shaped official-row A/B passes.
            cgan_complete_crown_ibp_root: false,
            // Publishing the cascade map to the `crown` input-split arm is
            // default-dark until a sealed official-budget row measures it.
            cgan_crown_root_map: false,
            // Dense-mode collection is an E5 experiment key: default-dark.
            cgan_collector_dense: false,
            // #cgan-unstable-rows: typed delivery of the existing sparse
            // ReLU-row collector gate. Default-dark until a sealed
            // official-budget cgan row measures it.
            cgan_unstable_rows: false,
            cgan_unstable_rows_max_frac: default_cgan_unstable_rows_max_frac(),
            // Narrow-target collection ceiling is default-dark: it changes
            // which targets the cascade spends its budget on, so it stays off
            // until a sealed official-budget row measures it.
            cgan_narrow_target_max_rows: None,
            // Child-side intermediate refinement is default-dark: it replaces a
            // map lookup with a per-child CROWN cascade, which is a large cost
            // change on every domain.
            cgan_child_interm_refine_max_secs: None,
            // #cgan-child-seed: all three child-cascade refinements are
            // default-dark; false/0/false is the historical serial, unseeded
            // prepass and the historical root-map-only fresh clip.
            cgan_child_interm_refine_seed_from_parent: false,
            cgan_child_interm_refine_parallel: 0,
            cgan_child_interm_refine_fresh_clip_map: false,
            // #cgan-child-zono: default-dark (enabled: false, budget 16384).
            cgan_child_zono_refresh: CganChildZonoRefreshConfig::default(),
            // Sparse optimization: focus on top 30% most influential alphas.
            // Reduces SPSA variance when perturbing fewer coordinates.
            sparse_ratio: 0.3,
            // Adaptive skip disabled by default: the upstream reference has
            // no depth gate — it relies on early_stop_patience to terminate stalled
            // optimization. The threshold of 20 silently prevented α-CROWN from running
            // on all production graph models (>20 ReLU nodes), producing IBP-identical
            // bounds. The retained threshold (20) only applies when explicitly enabled.
            adaptive_skip: false,
            adaptive_skip_depth_threshold: 20,
            // Pilot iteration disabled by default: the 1e-3 absolute threshold
            // exits too early for deep DAG networks (ResNet-2b) where gradient attenuation
            // makes first-iteration improvement tiny. The reference α,β-CROWN has no pilot
            // concept — it uses patience-based early stopping instead (early_stop_patience).
            adaptive_skip_pilot: false,
            // Require at least 1e-3 improvement from pilot to continue optimization.
            pilot_improvement_threshold: 1e-3,
            // Patience: 10 consecutive non-improving iterations before stopping.
            // Matches α,β-CROWN's early_stop_patience=10 (optimized_bounds.py:75-77).
            // Previously hardcoded to 3 — too aggressive for deep DAG networks.
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
            // One cumulative 25%-of-remaining reference-refresh pool with no
            // absolute ceiling. Presets may add a ceiling for categories where
            // refresh cost scales with a long official budget but does not
            // improve the retained bounds.
            reference_refresh_fraction: default_reference_refresh_fraction(),
            // Default OFF: byte-identical to the historical improved_output gate.
            joint_interm_alpha_every: 0,
            alpha_envelope_grad: false,
            reference_refresh_max_secs: None,
            // Preserve the historical forward-linear -> CROWN-IBP refusal
            // chain unless a measured category opts into the cheaper sound
            // endgame fallback.
            forward_linear_deadline_fallback_to_ibp: false,
            // Ordinary zero-iteration callers still consume the initial CROWN
            // evaluation. The graph-BaB bootstrap opts into skipping it only
            // for its fixed-intermediate collection route.
            skip_zero_iteration_collection_initial_bound: false,
            // No deadline by default (run all iterations)
            deadline: None,
            // No spec early-exit by default: warmup runs to the iteration/time cap
            // exactly as before. Set programmatically by the single-objective ReLU-split
            // warmup so it can stop once the root bound clears the threshold.
            spec_early_exit: None,
            spec_ascent: None,
            root_alpha_margin: false,
            alpha_zero_yield_frac: None,
            inert_ascent_exit_iters: None,
            // #vit-crown-interm: no cap on the fix_interm_bounds=false route.
            crown_ibp_reference_cap_secs: None,
            // #vit-margin-alpha: both default-dark; no shipped preset arms them.
            root_alpha_margin_gradient: false,
            matmul_alpha: false,
            // #vit-margin-alpha-complex: default-dark; no shipped preset arms it.
            margin_alpha_freeze_mul_reciprocal: false,
            // #vit-root-alpha: the certified ascent; no shipped preset arms V2.
            margin_alpha_search: MarginAlphaSearch::Ascent,
        }
    }
}

impl AlphaCrownConfig {
    /// Historical aggregate reference-refresh share.
    pub const DEFAULT_REFERENCE_REFRESH_FRACTION: f32 = 0.25;

    /// Lowest useful configured reference-refresh fraction. Matches the
    /// historical environment override parser.
    pub const MIN_REFERENCE_REFRESH_FRACTION: f32 = 0.01;

    /// Whether a configured aggregate refresh fraction is finite and within
    /// the supported scheduling range.
    pub fn reference_refresh_fraction_is_valid(fraction: f32) -> bool {
        fraction.is_finite() && (Self::MIN_REFERENCE_REFRESH_FRACTION..=1.0).contains(&fraction)
    }

    /// Whether a configured #alpha-zero-yield fraction is admissible. Matches
    /// the historical `NY_ALPHA_ZERO_YIELD_FRAC` parser exactly: finite,
    /// strictly positive, strictly below 0.9 (retiring the ascent at >=90% of
    /// its own window would be indistinguishable from the deadline it exists
    /// to preempt).
    pub fn alpha_zero_yield_frac_is_valid(fraction: f64) -> bool {
        fraction.is_finite() && fraction > 0.0 && fraction < 0.9
    }

    /// Whether a configured #inert-ascent-exit patience is admissible: at
    /// least one exactly-zero iteration must be observed before the exit can
    /// fire, so `0` is not a patience but a request to skip the ascent, which
    /// is what `iterations: 0` is for.
    pub fn inert_ascent_exit_iters_is_valid(iters: usize) -> bool {
        iters >= 1
    }

    /// Largest admissible `#cgan-child-seed` concurrency request
    /// (`cgan_child_interm_refine_parallel`). The runtime admission narrows
    /// any request further; this only rejects a value no host can honour.
    pub const MAX_CGAN_CHILD_INTERM_REFINE_PARALLEL: usize = 64;

    /// Whether a configured `#cgan-child-seed` concurrency request is
    /// admissible: `0` (serial) through [`Self::MAX_CGAN_CHILD_INTERM_REFINE_PARALLEL`].
    pub fn cgan_child_interm_refine_parallel_is_valid(width: usize) -> bool {
        width <= Self::MAX_CGAN_CHILD_INTERM_REFINE_PARALLEL
    }

    /// Largest admissible #vit-crown-interm reference cap, matching the
    /// historical collector-cap parser's ceiling.
    pub const MAX_CROWN_IBP_REFERENCE_CAP_SECS: u64 = 3_600;

    /// Whether a configured #vit-crown-interm reference cap is admissible:
    /// whole seconds in `1..=3600`. `0` would mean "never tighten anything",
    /// which is what `fix_interm_bounds: true` already is.
    pub fn crown_ibp_reference_cap_secs_is_valid(secs: u64) -> bool {
        (1..=Self::MAX_CROWN_IBP_REFERENCE_CAP_SECS).contains(&secs)
    }

    /// The effective #vit-crown-interm cap, or `None` when this config is not
    /// on the `fix_interm_bounds = false` route or carries no admissible cap.
    ///
    /// Fail-closed for programmatic callers: an inadmissible value resolves to
    /// "no cap", i.e. the historical uncapped route.
    #[must_use]
    pub fn crown_ibp_reference_cap(&self) -> Option<std::time::Duration> {
        if self.fix_interm_bounds {
            return None;
        }
        self.crown_ibp_reference_cap_secs
            .filter(|&secs| Self::crown_ibp_reference_cap_secs_is_valid(secs))
            .map(std::time::Duration::from_secs)
    }

    /// Resolve a directly-constructed or deserialized fraction fail-closed.
    ///
    /// Preset application rejects malformed values. This defensive fallback
    /// protects programmatic callers that construct the public config directly:
    /// an invalid value cannot disable or monopolize refresh scheduling.
    pub fn resolved_reference_refresh_fraction(&self) -> f32 {
        if Self::reference_refresh_fraction_is_valid(self.reference_refresh_fraction) {
            self.reference_refresh_fraction
        } else {
            default_reference_refresh_fraction()
        }
    }

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

    /// Check whether the deadline has been exceeded.
    ///
    /// Returns `true` if a deadline is set and the current time is past it.
    pub fn past_deadline(&self) -> bool {
        self.deadline.map(|d| Instant::now() >= d).unwrap_or(false)
    }
}

#[cfg(test)]
#[path = "alpha_config_tests.rs"]
mod tests;
