// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core `BetaCrownConfig` struct, `Default` impl, presets, and domain helper methods.

use std::time::{Duration, Instant};

use ny_core::{nan_propagating_max, NyError, Result};
use serde::{Deserialize, Serialize};

use crate::pgd_attack::{
    AdamClippingParams, PgdAlphaMode, PgdConfig, PgdInitialization, PgdOptimizer,
    GAMA_LAMBDA_DEFAULT,
};
use crate::AlphaCrownConfig;

use super::cut_config::{CutEvictionPolicy, CutScoreWeights};
use super::defaults::*;
use super::phase_budget::PhaseBudgetConfig;
use super::AdaptiveOptConfig;
use crate::beta_crown::branching::BranchingHeuristic;

/// Authority requested from a verification run.
///
/// `CertificateExport` is the fail-closed default: a proof lane that cannot be
/// represented by the current external certificate format must decline verdict
/// authority. `VerdictOnly` is an explicit request to produce only the runtime
/// verdict, as used by the scored VNN-COMP entry point.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationArtifactAuthority {
    /// The caller may consume a runtime verdict without an exported certificate.
    VerdictOnly,
    /// The caller requested certificate emission for any `Verified` verdict.
    #[default]
    CertificateExport,
}

/// Configuration for β-CROWN branch-and-bound search.
///
/// This pre-1.0 configuration surface grows as new proof policies acquire
/// typed authority. Downstream Rust callers should construct it from
/// [`Self::default`] and override selected fields instead of using exhaustive
/// struct literals; serialized configurations remain backward-compatible
/// through `#[serde(default)]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BetaCrownConfig {
    /// Runtime proof-artifact authority resolved by the frontend.
    ///
    /// This is deliberately not configurable through presets or serialized
    /// solver configuration. Frontends must derive it from their typed proof
    /// request after loading every preset; deserialization therefore restores
    /// the fail-closed `CertificateExport` default.
    #[serde(skip)]
    pub verification_artifact_authority: VerificationArtifactAuthority,
    /// Maximum number of domains to explore before giving up.
    #[serde(default = "default_max_domains")]
    pub max_domains: usize,
    /// Maximum number of domains to store in the domain queue simultaneously.
    ///
    /// When the queue exceeds this limit, the lowest-priority domains are evicted
    /// to prevent unbounded memory growth during BaB on hard verification instances.
    /// Set to 0 to disable the cap (unbounded queue, original behavior).
    ///
    /// Reference: Issue #2326 Finding 1
    #[serde(default = "default_max_queue_size")]
    pub max_queue_size: usize,
    /// Estimated payload cap in bytes for supported graph BaB frontiers.
    ///
    /// Coverage includes ordinary and precomputed ReLU-split heaps plus scalar
    /// GPU `DomainList` ReLU- and input-split routes. Heap routes use
    /// `estimate_graph_domain_bytes`; `DomainList` routes recompute their
    /// shared live-row census after every add. This is a logical-payload
    /// estimate, not an allocator-capacity, process-RSS, or device-memory
    /// limit. Grouped-disjunctive `DomainList` rejects a nonzero cap until its
    /// row sidecar is fully censused.
    ///
    /// When the resident queue exceeds the cap, the lowest-priority (least
    /// promising) domains are discarded. Evicted domains are unexplored and
    /// unverified, so the run is forced to `Unknown` rather than `Verified`
    /// (`GraphBabLifecycle::unresolved_due_to_eviction`); no bound is ever
    /// widened, so the result stays sound.
    ///
    /// `0` disables the cap and is the default, preserving uncapped queue
    /// semantics. A queue retains at least one domain for forward progress, so
    /// one oversized domain may exceed the requested estimate.
    ///
    /// Reference: #ml4acopf-bab-queue-mem
    #[serde(default)]
    pub max_queue_bytes: usize,
    /// Concrete engine timeout for the entire search.
    ///
    /// A zero duration expires immediately. Frontends that expose zero as an
    /// unbounded sentinel must translate it before constructing the verifier.
    #[serde(default = "default_timeout")]
    pub timeout: Duration,
    /// Maximum depth of the search tree (max number of splits).
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,
    /// Use α-CROWN optimization within each domain.
    #[serde(default = "default_use_alpha_crown")]
    pub use_alpha_crown: bool,
    /// Reuse forward-linear intermediate bounds in the final backward pass.
    ///
    /// Matches alpha-beta-CROWN `forward+backward` / `forward+crown`
    /// semantics: compute intermediate graph bounds with forward symbolic
    /// relaxation, then keep the final objective solve on backward CROWN.
    #[serde(default)]
    pub use_forward_bounds: bool,
    /// Optimize alpha slopes independently for each disjunct in OR-spec
    /// properties. Reference: alpha-beta-CROWN `optimize_disjuncts_separately`.
    #[serde(default)]
    pub optimize_disjuncts_separately: bool,
    /// Hard wall-clock cap (seconds) on the ROOT alpha-CROWN warmup, including
    /// the intermediate-bound collection that runs inside it.
    ///
    /// `None` (default) leaves the warmup bounded only by the phase budget,
    /// which is what it did historically: on a deep conv DAG it then expands to
    /// consume essentially the whole initial-bounds slice and BaB inherits
    /// whatever is left. That is a bad trade when the root bound is already
    /// good enough to close: measured on cifar100_2024
    /// CIFAR100_resnet_medium_prop_idx_9694 at the OFFICIAL 100s budget, the
    /// margin-row BaB needs ~26s to reach UNSAT, and it only gets that if the
    /// root is capped. Capped root (40s) + a reclaimed disjunctive-PGD slice
    /// turns that instance from `timeout` into `unsat` at 100s; NEITHER change
    /// alone is sufficient (both measured `timeout` in isolation).
    ///
    /// Deadlines only schedule work — every path falls back to sound reference
    /// bounds on expiry — so this knob carries no soundness obligation.
    /// `NY_ROOT_ALPHA_CAP_SECS` still overrides it.
    #[serde(default)]
    pub root_alpha_cap_secs: Option<f64>,
    /// Retain a completed, certified DAG-alpha artifact when the root warmup's
    /// local phase cap expires, so the multi-objective root can re-evaluate it
    /// under the still-live outer verifier deadline.
    ///
    /// This is scheduling state only: the retained alpha/reference map cannot
    /// grant a verdict, and every objective is evaluated by the normal certified
    /// root path. Default false keeps unqualified presets on their historical
    /// deadline mapping. An absent `NY_ROOT_ALPHA_PHASE_CHECKPOINT` inherits
    /// this setting, literal `1` force-arms, and every other present value
    /// force-disarms the typed policy.
    #[serde(default)]
    pub root_alpha_phase_checkpoint: bool,
    /// Number of additional exact full-`C` root margin-Adam iterations.
    ///
    /// Zero is the default-dark historical behavior. A positive value arms the
    /// score-bearing atomic root-`C` route directly (without arming the DAG
    /// identity pre-loop), re-evaluating the complete source-ordered `C` matrix
    /// after every proposal. The resource bound is deliberately small: at most
    /// [`ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS`] whole-`C` proposals may run.
    #[serde(default)]
    pub atomic_root_c_margin_iterations: usize,
    /// Use CROWN-IBP for tighter intermediate bounds.
    /// When enabled, intermediate bounds are computed using CROWN backward
    /// propagation and intersected with IBP bounds, producing ~66% tighter bounds
    /// than standard CROWN. More expensive but significantly improves verification.
    #[serde(default)]
    pub use_crown_ibp: bool,
    /// Maximum number of sequential CROWN-IBP partial nodes before falling back
    /// to plain IBP intermediate bounds.
    ///
    /// Counts only the layer outputs that would trigger an expensive partial
    /// CROWN pass, not every layer in the sequential network.
    #[serde(default)]
    pub max_crown_ibp_nodes: Option<usize>,
    /// Floor (seconds) for CROWN-IBP collectors' per-node time budget
    /// (#4413, #cgan-bn11-budget). A node whose share
    /// falls below the floor is skipped to sound IBP. `None` (default) keeps
    /// the built-in 2.0 s constant — behavior is unchanged when unset.
    #[serde(default)]
    pub crown_ibp_per_node_floor_secs: Option<f64>,
    /// Explicit base cap (seconds) on CROWN-IBP collectors' per-node time
    /// budget (#4413, #cgan-bn11-budget). `None` derives an adaptive cap from
    /// the remaining collection budget (25%, clamped to 12–600 seconds).
    /// Explicit caps are dimension-scaled above 28,800 rows. Purely a
    /// time-vs-tightness policy knob: any value is sound.
    #[serde(default)]
    pub crown_ibp_per_node_cap_secs: Option<f64>,
    /// Maximum number of backward layers/nodes for fixed-slope CROWN.
    ///
    /// When set, CROWN backward stops after propagating through this many
    /// output-adjacent layers or graph nodes and concretizes the remaining
    /// frontier with sound forward bounds. This mirrors alpha-beta-CROWN's
    /// fixed-intermediate-bound truncation during BaB refinement.
    #[serde(default)]
    pub crown_backward_layers: Option<usize>,
    /// α-CROWN configuration (if use_alpha_crown is true).
    #[serde(default)]
    pub alpha_config: AlphaCrownConfig,
    /// Branching heuristic for selecting neurons to split.
    #[serde(default)]
    pub branching_heuristic: BranchingHeuristic,
    /// Number of input dimensions to split per input-splitting step.
    ///
    /// Higher values create 2^k child domains per split. Default: 1.
    #[serde(default = "default_input_split_depth")]
    pub input_split_depth: usize,
    /// Coefficient clamp threshold for input-splitting SB scoring.
    /// Matches alpha-beta-CROWN `sb_coeff_thresh` (default: 1e-3).
    #[serde(default = "default_input_split_coeff_thresh")]
    pub input_split_coeff_thresh: f32,
    /// Per-sub-domain α refinement iterations in the input-split BaB loop.
    ///
    /// When > 0 AND `use_alpha_crown` is enabled, each sub-domain warm-starts
    /// from its parent's optimized alphas and re-optimizes them for this many
    /// SPSA iterations against the sub-domain's tighter box (with
    /// `fix_interm_bounds=true`, skipping the O(N²) intermediate CROWN pass).
    /// The refined alphas are saved onto both child domains. This produces much
    /// tighter per-domain bounds → fewer splits.
    ///
    /// Default 0 keeps the historical single frozen-alpha bound computation per
    /// domain (root-optimized alphas threaded in, zero gradient steps), and the
    /// warm-start branch is never taken. The reordered production path is
    /// unchanged. Eager grouped-disjunctive screening also retains the sound
    /// component-wise lower floors from its parent and clip recomputes, so it may
    /// finish an already jointly verified domain earlier even at this default.
    ///
    /// Matches alpha-beta-CROWN `solver.alpha-crown.input_split_alpha_iteration`
    /// (reference default 5, `input_split/bounding.py:90-179`).
    #[serde(default = "default_input_split_alpha_iteration")]
    pub input_split_alpha_iteration: usize,
    /// Learning rate for per-sub-domain α refinement (see
    /// `input_split_alpha_iteration`). Only used when that knob is > 0.
    ///
    /// Matches alpha-beta-CROWN `solver.alpha-crown.input_split_lr_alpha`
    /// (default 0.05).
    #[serde(default = "default_input_split_lr_alpha")]
    pub input_split_lr_alpha: f32,
    /// Bonus score for bounds that touch zero during `sb_sum` input split scoring.
    /// Matches alpha-beta-CROWN `touch_zero_score` (default: 0.0 = disabled).
    #[serde(default = "default_input_split_touch_zero_score")]
    pub input_split_touch_zero_score: f32,
    /// Margin weight for SB scoring: scales the per-spec verification margin.
    ///
    /// Uses lower bounds in lower-bound mode and upper bounds in upper-bound
    /// mode so the heuristic stays aligned with the active verification target.
    /// When positive, dimensions that help nearly verified specs score higher.
    /// Matches alpha-beta-CROWN `sb_margin_weight` (default: 1.0).
    ///
    /// Reference: alpha-beta-CROWN `branching_heuristics.py:84-89`
    #[serde(default = "default_input_split_sb_margin_weight")]
    pub input_split_sb_margin_weight: f32,
    /// Sum across specs instead of taking max for SB scoring.
    ///
    /// When true, dimension scores are summed across all specification rows.
    /// When false (default), the max across specs is used.
    /// Matches alpha-beta-CROWN `sb_sum` (default: false).
    ///
    /// Reference: alpha-beta-CROWN `branching_heuristics.py:79-81`
    #[serde(default)]
    pub input_split_sb_sum: bool,
    /// Primary spec index for SB scoring (use single spec instead of max/sum).
    ///
    /// When set, only the specified spec row is used for scoring.
    /// Matches alpha-beta-CROWN `sb_primary_spec` (default: None = all specs).
    ///
    /// Reference: alpha-beta-CROWN `branching_heuristics.py:91-93`
    #[serde(default)]
    pub input_split_sb_primary_spec: Option<usize>,
    /// Number of candidate neurons to evaluate for FSB/kFSB branching.
    ///
    /// Higher values can reduce domain count but increase per-domain overhead.
    /// Default: 8 (ny default).
    #[serde(default = "default_fsb_candidates")]
    pub fsb_candidates: usize,
    /// Reduce operation for kFSB branching (how to combine branch scores).
    /// - Min: conservative (worst-case) - default
    /// - Max: optimistic (best-case)
    /// - Mean: balanced (average)
    #[serde(default)]
    pub kfsb_reduce_op: KfsbReduceOp,
    /// Bounded, branch-specific depth-2 lookahead advice for the
    /// multi-objective graph kFSB lane.
    ///
    /// This policy is default-off. When armed, it only ranks which already
    /// constructed first-level split to commit; private child-bound maps and
    /// second-level scores never become proof authority. Phase 1 is
    /// lower-bound-only and fails closed during upper-bound verification so
    /// the historical raw-lower objective selection remains unchanged. An
    /// admitted typed budget has priority over legacy M27/f64 observers for
    /// that wave; their advice-only one-shots remain available for a later
    /// wave instead of suppressing typed selection or sharing its deadline.
    #[serde(default)]
    pub depth_two_branch_lookahead: DepthTwoBranchLookaheadConfig,
    /// Learning rate for β parameter optimization.
    pub beta_lr: f32,
    /// Number of optimization iterations per domain (applies to both α and β).
    pub beta_iterations: usize,
    /// Minimum improvement to continue optimization.
    pub beta_tolerance: f32,
    /// Break after this many consecutive iterations without a better lower bound.
    ///
    /// Matches alpha-beta-CROWN `early_stop_patience=10`
    /// (`optimized_bounds.py:75-77`). Set to `usize::MAX` to disable the
    /// patience guard while keeping the gradient-convergence exit active.
    #[serde(default = "default_early_stop_patience")]
    pub early_stop_patience: usize,
    /// Number of β optimization iterations to run on the root domain before BaB.
    /// This amortizes the optimization cost across all descendant domains via warmup.
    /// Set to 0 to disable root-level optimization (default: 20).
    #[serde(default = "default_root_beta_iterations")]
    pub root_beta_iterations: usize,
    /// Maximum depth at which to run per-domain β optimization.
    /// Domains deeper than this use inherited β values without further optimization.
    /// Set to 0 to disable per-domain optimization entirely (rely only on root + warmup).
    /// Default: 3 (optimize root and first 3 levels of splits).
    #[serde(default = "default_beta_max_depth")]
    pub beta_max_depth: usize,
    /// Use analytical β gradients for DAG networks instead of SPSA.
    /// Analytical gradients are computed from the A matrices stored during
    /// CROWN backward propagation, which is ~3x faster than SPSA (1 pass vs 3 passes per iteration).
    /// Default: true (use analytical gradients when available).
    #[serde(default = "default_use_analytical_beta_gradients")]
    pub use_analytical_beta_gradients: bool,
    /// Learning rate for α parameter optimization (when use_alpha_crown is true).
    pub alpha_lr: f32,
    /// Use momentum for α updates (inherits from alpha_config.momentum if true).
    pub alpha_momentum: bool,
    /// Number of domains to process in parallel (batch size).
    /// Set to 1 for sequential processing, or higher for parallel.
    pub batch_size: usize,
    /// Maximum number of spec rows to handle in one root build batch.
    ///
    /// Mirrors alpha-beta-CROWN `solver.build_batch_size` and is primarily used
    /// by large graph input-split warmup on multi-spec properties such as
    /// nn4sys `mscn_2048*`, where a single dense spec-matrix build can consume
    /// the full timeout budget before BaB begins.
    ///
    /// `None` keeps the current all-specs-at-once behavior.
    #[serde(default)]
    pub build_batch_size: Option<usize>,
    /// Use parallel child domain creation (both branches computed in parallel).
    pub parallel_children: bool,
    /// Use adaptive learning rates (Adam-style optimizer).
    /// When enabled, learning rates are automatically adjusted per-parameter
    /// based on gradient history, improving convergence on diverse problems.
    pub use_adaptive: bool,
    /// Configuration for adaptive optimizer (when use_adaptive is true).
    pub adaptive_config: AdaptiveOptConfig,
    // --- GCP-CROWN: Cutting Plane Configuration ---
    /// Enable GCP-CROWN cutting planes.
    /// When enabled, verified subdomains generate cuts that tighten bounds
    /// for remaining domains.
    #[serde(default)]
    pub enable_cuts: bool,
    /// Convolution backward mode for CROWN propagation.
    /// Auto (default): Patches when cuts disabled, Matrix when cuts enabled.
    /// Reference: alpha-beta-CROWN `general.conv_mode` (`abcrown.py:228-231`).
    #[serde(default)]
    pub conv_mode: ConvMode,
    /// Maximum number of cutting planes to retain.
    /// More cuts = tighter bounds but more computation per domain.
    #[serde(default = "default_max_cuts")]
    pub max_cuts: usize,
    /// Minimum depth of verified domain to generate a cut.
    /// Deeper domains produce more specific cuts.
    #[serde(default = "default_min_cut_depth")]
    pub min_cut_depth: usize,
    /// Enable near-miss cut generation.
    /// When enabled, cuts are also generated from domains where the lower bound
    /// is within `near_miss_margin` of the threshold, even if not verified.
    /// This can help prune similar regions that are "almost verified".
    #[serde(default)]
    pub enable_near_miss_cuts: bool,
    /// Margin for near-miss cut generation (as fraction of threshold).
    /// Only used when `enable_near_miss_cuts` is true.
    /// Default: 0.1 (10% of threshold, or absolute 0.1 if threshold is 0)
    #[serde(default = "default_near_miss_margin")]
    pub near_miss_margin: f32,
    /// Enable proactive cut generation (BICCOS-lite).
    /// When enabled, cuts are generated for unstable ReLUs BEFORE BaB starts,
    /// rather than waiting for domains to verify. This helps on hard instances
    /// where no domains verify initially (chicken-and-egg problem).
    ///
    /// The proactive cuts encode pairwise neuron implications based on initial bounds.
    #[serde(default)]
    pub enable_proactive_cuts: bool,
    /// Maximum number of proactive cuts to generate.
    /// More cuts = potentially tighter bounds but more computation.
    #[serde(default = "default_max_proactive_cuts")]
    pub max_proactive_cuts: usize,
    /// Enable BICCOS constraint strengthening for verified-domain cuts.
    /// When enabled, cuts are strengthened by dropping low-influence constraints
    /// (based on neuron influence scores) and re-verifying the reduced domain.
    #[serde(default)]
    pub enable_biccos_constraint_strengthening: bool,
    /// Drop ratio for BICCOS constraint strengthening.
    /// Uses the quantile of influence scores; higher values drop more constraints.
    #[serde(default = "default_biccos_drop_ratio")]
    pub biccos_drop_ratio: f32,
    /// Optimization interval for cut lambda parameters (domains between updates).
    /// Lambda optimization runs every N domains explored during BaB.
    /// Lower values update more frequently (better convergence, more overhead).
    #[serde(default = "default_lambda_opt_interval")]
    pub lambda_opt_interval: usize,
    /// Learning rate for cut lambda Adam optimization.
    /// Overrides `adaptive_config.lr_lambda` specifically for the BaB lambda path.
    /// Default: 0.05 (matches alpha-beta-CROWN lambda LR).
    #[serde(default = "default_lambda_lr")]
    pub lambda_lr: f32,
    /// Cut eviction policy for bounded cut pools.
    #[serde(default)]
    pub cut_eviction_policy: CutEvictionPolicy,
    /// Iteration threshold for stale cuts.
    #[serde(default = "default_cut_stale_iters")]
    pub cut_stale_iters: usize,
    /// Iteration threshold for hard-stale cuts (aggressive eviction).
    #[serde(default = "default_cut_hard_stale_iters")]
    pub cut_hard_stale_iters: usize,
    /// Lambda threshold below which stale cuts are evicted.
    #[serde(default = "default_cut_lambda_min")]
    pub cut_lambda_min: f32,
    /// Maximum fraction of proactive cuts allowed in the pool.
    #[serde(default = "default_cut_proactive_fraction")]
    pub cut_proactive_fraction: f32,
    /// Scoring weights for utility-weighted eviction.
    #[serde(default)]
    pub cut_score_weights: CutScoreWeights,
    // --- BICCOS Cold-Start Gating ---
    /// Enable cold-start gating for BICCOS cut generation.
    /// When enabled, cuts are withheld until early BaB statistics indicate
    /// verified domains or bound gains are sufficient to make cuts effective.
    #[serde(default)]
    pub enable_biccos_cold_start: bool,
    /// Minimum verified domains before enabling cuts.
    #[serde(default = "default_biccos_min_verified")]
    pub biccos_min_verified: usize,
    /// Minimum verified domain rate (per iteration) before enabling cuts.
    #[serde(default = "default_biccos_min_verified_rate")]
    pub biccos_min_verified_rate: f32,
    /// Sliding window size for verified-rate computation.
    #[serde(default = "default_biccos_verified_rate_window")]
    pub biccos_verified_rate_window: usize,
    /// Minimum cuts generated (e.g., via MTS) before enabling cuts.
    #[serde(default = "default_biccos_min_cuts")]
    pub biccos_min_cuts: usize,
    /// Minimum average bound gain per split before enabling cuts.
    #[serde(default = "default_biccos_min_bound_gain")]
    pub biccos_min_bound_gain: f32,
    /// Sliding window size for bound-gain computation.
    #[serde(default = "default_biccos_bound_gain_window")]
    pub biccos_bound_gain_window: usize,
    /// Maximum number of cold-start iterations before declaring exhaustion.
    #[serde(default = "default_biccos_cold_max_iters")]
    pub biccos_cold_max_iters: usize,
    /// Maximum number of iterations to keep cut generation enabled.
    #[serde(default = "default_biccos_cut_window")]
    pub biccos_cut_window: usize,
    /// Minimum cut yield before disabling new cut generation.
    #[serde(default = "default_biccos_min_cut_yield")]
    pub biccos_min_cut_yield: f32,
    /// Sliding window size for cut-yield computation.
    #[serde(default = "default_biccos_cut_yield_window")]
    pub biccos_cut_yield_window: usize,
    /// Number of low-yield windows before disabling cut generation.
    #[serde(default = "default_biccos_cut_yield_patience")]
    pub biccos_cut_yield_patience: usize,
    // --- Property Direction ---
    /// Verify upper bound instead of lower bound.
    ///
    /// When `false` (default): verifies output > threshold (lower_bound > threshold)
    /// When `true`: verifies output < threshold (upper_bound < threshold)
    ///
    /// Use `true` for VNNLIB constraints like `Y >= c` (unsafe region), where
    /// proving safety requires proving the upper bound is below the threshold.
    #[serde(default)]
    pub verify_upper_bound: bool,
    // --- PGD Attack for Counterexample Finding ---
    /// Enable PGD attack to find concrete counterexamples.
    /// When enabled and verification is inconclusive, PGD attack is run to
    /// try to find a concrete input that violates the property.
    #[serde(default)]
    pub enable_pgd_attack: bool,
    /// Number of random restarts for PGD attack.
    #[serde(default = "default_pgd_restarts")]
    pub pgd_restarts: usize,
    /// Number of gradient steps per restart.
    #[serde(default = "default_pgd_steps")]
    pub pgd_steps: usize,
    /// Restart PGD when a projected step leaves the point unchanged (#4278).
    ///
    /// Reference: alpha-beta-CROWN `pgd_restart_when_stuck` in
    /// `abcrown_all_params.yaml:279` and `acasxu.yaml:20`.
    #[serde(default)]
    pub pgd_restart_when_stuck: bool,
    /// PGD initialization strategy (#1449).
    ///
    /// `Uniform` (default): standard random sampling.
    /// `Osi`: Output Specification Initialization for diverse restarts.
    /// Reference: alpha-beta-CROWN `attack_mode: diversed_PGD` →
    /// `initialization = "osi"` in `attack_interface.py:29-35`.
    #[serde(default)]
    pub pgd_initialization: PgdInitialization,
    /// Number of OSI initialization steps (#1449). Default: 20.
    #[serde(default = "default_pgd_osi_steps")]
    pub pgd_osi_steps: usize,
    /// Enable the GAMA guidance loss for PGD attack steps (#1449).
    ///
    /// Set by `attack_mode: diversed_GAMA_PGD` (together with OSI
    /// initialization). Relational-target attack steps then ascend
    /// `softmax_margin + λ·‖P − softmax‖²` with λ annealed linearly from
    /// [`GAMA_LAMBDA_DEFAULT`] to 0 (Sriramanan et al., NeurIPS 2020) —
    /// designed for adversarially-trained networks whose raw-margin gradients
    /// are masked. Attack-only: candidates are re-validated before any `sat`,
    /// so this cannot affect soundness.
    /// Reference: alpha-beta-CROWN `attack_mode: diversed_GAMA_PGD` →
    /// `GAMA_loss=True` in `attack_interface.py:29-35`.
    #[serde(default)]
    pub pgd_gama: bool,
    /// PGD optimizer strategy (#4277).
    #[serde(default)]
    pub pgd_optimizer: PgdOptimizer,
    /// PGD alpha/step-size policy (#4277).
    #[serde(default)]
    pub pgd_alpha_mode: PgdAlphaMode,
    /// Per-step exponential decay for the PGD/Adam learning rate.
    ///
    /// Maps to `AdamClippingParams::lr_decay`. Lower values shrink the step
    /// size faster across PGD steps. Pure attack tuning: PGD only searches for
    /// counterexamples that are re-validated before being emitted as `sat`,
    /// so this cannot affect soundness.
    ///
    /// Reference: alpha-beta-CROWN `attack.pgd_lr_decay` /
    /// `attack_pgd.py:255` (`ExponentialLR(opt, lr_decay)`).
    #[serde(default = "default_pgd_lr_decay")]
    pub pgd_lr_decay: f32,
    /// Straight-through-estimator surrogate gradient for Sign layers during
    /// ATTACK gradient estimation (#surrogate-sign, preset key
    /// `attack.surrogate_sign_gradient`). Maps to
    /// `PgdConfig::surrogate_sign_gradient`. Attack-only: candidates are
    /// re-validated before any `sat`, so this cannot affect soundness.
    #[serde(default)]
    pub pgd_surrogate_sign_gradient: bool,
    /// Dense deterministic grid sweep over low-effective-dimension input
    /// boxes as a pre-PGD attack phase (#dense-sweep, preset key
    /// `attack.dense_low_dim_sweep`). Maps to
    /// `PgdConfig::dense_low_dim_sweep`. Attack-only.
    #[serde(default)]
    pub pgd_dense_low_dim_sweep: bool,
    /// Maximum number of nonzero-width input dims for the dense sweep to run
    /// (#dense-sweep). Default: 3.
    #[serde(default = "default_pgd_dense_sweep_max_dims")]
    pub pgd_dense_sweep_max_dims: usize,
    /// Total forward-evaluation budget for the dense sweep (#dense-sweep).
    /// Default: 32768 (a 128×128 initial grid plus refinement for 2 dims).
    #[serde(default = "default_pgd_dense_sweep_points")]
    pub pgd_dense_sweep_points: usize,
    // --- Relaxed Clipping (Clip-and-Verify) ---
    /// Enable relaxed clipping to tighten input bounds using CROWN constraints.
    ///
    /// When enabled, after each input split, the child domain's input bounds are
    /// tightened using closed-form 1D updates based on the CROWN linear constraints.
    /// This can significantly reduce the number of domains needed for verification.
    ///
    /// Requires `branching_heuristic` to be `InputSplit` to have effect.
    ///
    /// Reference: Wei et al., "Clip and Verify" (arXiv:2512.11087)
    #[serde(default)]
    pub enable_relaxed_clip: bool,
    /// Number of relaxed clipping iterations per split.
    ///
    /// More iterations can produce tighter bounds but with diminishing returns.
    /// Typical values: 1-3 iterations.
    /// Default: 1 (matches baseline)
    #[serde(default = "default_relaxed_clip_iterations")]
    pub relaxed_clip_iterations: usize,
    /// Input-domain clipping algorithm type.
    ///
    /// - `Relaxed` (default): Closed-form 1D updates, fast but axis-aligned.
    /// - `Complete`: LP-optimal via Lagrangian dual coordinate ascent, tighter
    ///   bounds by accounting for cross-constraint dependencies.
    ///
    /// Only has effect when `enable_relaxed_clip` is true.
    ///
    /// Reference: alpha-beta-CROWN `clip_input_domain.clip_type`
    #[serde(default)]
    pub input_clip_type: InputClipType,
    /// Default-off exact-domain Clip-and-Verify for batch-stack-unsafe
    /// multi-clause input splitting.
    ///
    /// Unlike the legacy post-split clip, this route computes one
    /// non-domain-stacked full-spec CROWN result on the exact popped domain,
    /// clips that same domain before choosing its split, drops the fresh planes,
    /// and queues children without any parent planes. It is deliberately
    /// restricted by [`Self::validate`] to the
    /// reordered + IBP-enhanced relaxed-clip route; the graph dispatcher further
    /// requires a batch-stack-unsafe graph. No shipped preset enables it.
    #[serde(default)]
    pub input_split_fresh_domain_clip: bool,
    /// Neuron selection ratio for complete clipping.
    ///
    /// **UNIMPLEMENTED**: Config is parsed and stored for preset compatibility
    /// but no engine code reads this field yet. See design doc Step 7
    /// (`designs/2026-03-10-issue-523-complete-clipping.md`).
    ///
    /// When `input_clip_type == Complete`, intended to control what fraction of
    /// unstable neurons receive constrained concretization (LP-optimal bounds).
    /// - `-1.0` (default): Disabled, apply to all neurons.
    /// - `[0.0, 1.0]`: Apply to that fraction of unstable neurons per layer,
    ///   selected by largest uncertainty (upper - lower gap).
    ///
    /// Reference: alpha-beta-CROWN `clip_neuron_selection_value`
    #[serde(default = "default_clip_neuron_selection_ratio")]
    pub clip_neuron_selection_ratio: f32,
    // --- Intermediate Domain Clipping (clip_interm_domain) ---
    /// Enable intermediate domain clipping to tighten intermediate neuron bounds.
    ///
    /// When enabled, uses split-derived linear constraints in input space to tighten
    /// intermediate (pre-activation) bounds via constrained concretization. This can
    /// significantly reduce the number of unstable neurons and improve verification.
    ///
    /// Requires ReLU branching (not input split) and split history with constraints.
    ///
    /// Reference: alpha-beta-CROWN `clip_interm_domain` feature
    /// (`complete_verifier/domain_clipper.py`)
    #[serde(default)]
    pub enable_clip_interm_domain: bool,
    /// Number of objective neurons per layer to tighten with clip_interm_domain.
    ///
    /// Higher values tighten more neurons but increase computation.
    /// Default: 3. Scored presets may opt into alpha-beta-CROWN's larger
    /// `clip_topk_objective` value explicitly.
    #[serde(default = "default_clip_interm_topk")]
    pub clip_interm_topk: usize,
    /// Apply clip_interm_domain during α-CROWN optimization passes.
    ///
    /// Baseline: clip_n_verify.clip_interm_domain.clip_in_alpha_crown.
    /// Default: false (maintains existing behavior unless explicitly enabled).
    #[serde(default)]
    pub clip_in_alpha_crown: bool,
    /// Prune infeasible domains detected during activation-space clipping.
    ///
    /// Baseline: clip_n_verify.prune (domain_clipper).
    /// Default: false. The certificate-backed batched production path
    /// currently quarantines pruning and rejects `true` during validation;
    /// Complete Clip may tighten bounds but cannot mint infeasibility yet.
    #[serde(default)]
    pub clip_interm_prune: bool,
    /// Use final-layer constraints when pruning clipped domains.
    ///
    /// Baseline: clip_n_verify.final_layer (domain_clipper).
    /// Default: false.
    #[serde(default)]
    pub clip_interm_use_final_layer: bool,
    // --- Batched Intermediate Transfer (interm_transfer) ---
    /// Enable static intermediate bound transfer in batched domains.
    ///
    /// When enabled, batched domain storage captures static intermediate bounds
    /// and unstable masks for reuse across domain batches.
    #[serde(default)]
    pub enable_interm_transfer: bool,
    // --- Root intermediate CROWN CUDA factory fallback ---
    /// Allow root intermediate CROWN passes to obtain a sound CUDA engine from
    /// the process factory when the caller did not provide a usable local
    /// engine.
    ///
    /// Default false keeps existing configurations unchanged. The exact
    /// `NY_ROOT_INTERM_CUDA_FACTORY` environment override remains available:
    /// absent inherits this typed value, `1` forces on, and every other present
    /// value forces off.
    #[serde(default)]
    pub root_interm_cuda_factory: bool,
    // --- Post-root multi-objective CUDA factory engine handoff ---
    /// Allow multi-objective graph BaB to reuse an already-materialized sound
    /// CUDA engine after root evaluation when the caller supplied no engine.
    ///
    /// The handoff never initializes or waits for the process factory. It also
    /// requires the selected engine to advertise deadline-safe support for the
    /// complete post-root execution surface (including generic GEMM), plus
    /// verdict-grade GPU CROWN soundness and cooperative cancellation. Default
    /// false preserves the historical post-root sequential route. The exact
    /// `NY_MO_CUDA_FACTORY_ENGINE_HANDOFF` override is resolved after root:
    /// absent inherits this value, literal `1` enables, and every other present
    /// value disables.
    #[serde(default)]
    pub mo_cuda_factory_engine_handoff: bool,
    // --- Post-root bounded CUDA β shared-executor activation ---
    /// Allow multi-objective graph BaB to enter its shared executor through a
    /// local deadline-aware CPU GEMM facade when no caller/post-root engine is
    /// available and an already-materialized sound CUDA backend exposes only
    /// the audited call-local bounded β-CROWN surface.
    ///
    /// CUDA is never handed to generic shared-executor work. Default false
    /// preserves the historical sequential route. The exact
    /// `NY_MO_CUDA_BOUNDED_SHARED_EXECUTOR` override is resolved after root:
    /// absent inherits this value, literal `1` enables, and every other present
    /// value disables.
    #[serde(default)]
    pub mo_cuda_bounded_shared_executor: bool,
    // --- Root dense-head intermediate CROWN (#cifar-head-crown) ---
    /// Run one bounded heuristic-slope CROWN backward for each dense-fed ReLU
    /// pre-activation at the graph root and shrink-intersect the certified box
    /// into the frozen intermediate bounds inherited by BaB.
    ///
    /// This is intentionally scoped structurally (Linear/Gemm -> ReLU), rather
    /// than by exporter-specific node names. Default false keeps every benchmark
    /// unchanged unless a measured preset opts in. `NY_ROOT_CROWN_INTERM=1/0`
    /// remains an explicit force-on/force-off override for A/B and rollback.
    #[serde(default)]
    pub root_crown_interm_dense_head: bool,
    /// Wall-clock cap in seconds for the one-time root dense-head CROWN pass.
    /// Zero skips the pass. A global verifier deadline always takes precedence.
    #[serde(default = "default_root_crown_interm_max_secs")]
    pub root_crown_interm_max_secs: u64,
    /// Maximum number of elements in a selected dense-head pre-activation.
    /// Zero selects no targets. `NY_ROOT_CROWN_INTERM_MAXDIM` overrides for A/B.
    #[serde(default = "default_root_crown_interm_max_dim")]
    pub root_crown_interm_max_dim: usize,
    // --- Root sparse crossing-row intermediate CROWN (#root-sparse-interm-crown) ---
    /// Tighten structurally eligible convolutional ReLU pre-activations at the
    /// graph root by seeding only their widest crossing rows into the certified
    /// sound GPU CROWN fold, then shrink-intersecting the resulting bounds.
    ///
    /// Dense/Gemm heads are deliberately excluded because
    /// `root_crown_interm_dense_head` handles them with a full identity seed.
    /// Default false keeps unmeasured presets byte-identical. The
    /// `NY_ROOT_SPARSE_INTERM_CROWN=1/0` override is retained for sealed A/B and
    /// emergency rollback.
    #[serde(default)]
    pub root_sparse_interm_crown: bool,
    /// One-time wall-clock cap. Zero skips the pass; the verifier deadline wins.
    #[serde(default = "default_root_sparse_interm_crown_max_secs")]
    pub root_sparse_interm_crown_max_secs: u64,
    /// Maximum flattened pre-activation width admitted structurally.
    #[serde(default = "default_root_sparse_interm_crown_max_dim")]
    pub root_sparse_interm_crown_max_dim: usize,
    /// Maximum widest crossing rows seeded per target.
    #[serde(default = "default_root_sparse_interm_crown_max_rows")]
    pub root_sparse_interm_crown_max_rows: usize,
    /// Maximum eligible targets, processed deepest-first.
    #[serde(default = "default_root_sparse_interm_crown_max_targets")]
    pub root_sparse_interm_crown_max_targets: usize,
    // --- Row-chunked comprehensive sound-GPU root intermediate sweep ---
    /// Run the comprehensive all-target sound-GPU root intermediate sweep, which
    /// injects identity rows at every eligible ReLU pre-activation depth in ONE
    /// atomic backend transaction and shrink-intersects the result.
    ///
    /// DELIVERY (this key exists because of `measured_gate_delivery.rs`): the
    /// scored entry point exports exactly one `NY_*` variable, so an env-only gate
    /// is dead in competition no matter what it measured. The measurement below
    /// was taken through `NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN=1`; this typed key
    /// is how it actually reaches a scored run. The env lever is retained as a
    /// force-on/force-off override for A/B and rollback.
    ///
    /// MEASURED (cifar100 resnet_medium, official 100 s budget, scored config):
    /// with `root_comprehensive_gpu_interm_chunks` > 1 the root objective census
    /// goes from 0/99 to 92/99 on the best row and 8 of 14 sampled timeout rows
    /// gain, worst-objective margin -18.42 -> -1.09. No row converted yet.
    /// Default false keeps unmeasured presets byte-identical.
    #[serde(default)]
    pub root_comprehensive_gpu_interm: bool,
    /// `#bab-floor`: BaB's guaranteed share of the multi-objective root window,
    /// subtracted BEFORE any root phase sizes itself.
    ///
    /// DELIVERY: the scored entry point exports exactly one `NY_*`, so the env
    /// lever these three were measured through is dead in competition. These
    /// typed keys are how a search result reaches a scored run.
    ///
    /// `None` (the default) means no arbitration exists and every root phase
    /// keeps the deadline it has today, so an unmeasured preset stays
    /// byte-identical. `NY_BAB_RESERVE_FRAC` overrides in BOTH directions.
    #[serde(default)]
    pub root_bab_reserve_frac: Option<f64>,
    /// `#bab-floor`: the root objective pass's share, subtracted after
    /// `root_bab_reserve_frac`. Read only when that reservation is armed —
    /// without a share of its own the pass starves and the BaB reserve behind
    /// it is unreachable. `NY_ROOT_SPEC_FRAC` overrides.
    #[serde(default)]
    pub root_spec_frac: Option<f64>,
    /// `#bab-floor`: the bootstrap ascent's share, min-composed onto
    /// `root_alpha_cap_secs`. That cap is a FIXED 40 s — 51% of the BaB slice
    /// at 100 s and 4% at 1200 s — so it is the one claimant whose cost does
    /// not scale with the window. `NY_ROOT_ALPHA_FRAC` overrides.
    #[serde(default)]
    pub root_alpha_frac: Option<f64>,
    /// How many DISJOINT row windows the comprehensive sweep may accumulate.
    ///
    /// The sweep is hard-capped in rows-per-target by device memory (the backend
    /// declines beyond its class budget), which is far below the coverage the root
    /// needs. Running the same bounded sweep repeatedly over disjoint windows
    /// keeps peak device memory at one chunk while coverage grows with the time
    /// available. Sound by construction: every sweep is atomic over its own
    /// window, every commit is a shrink-only intersect, and all windows are cut
    /// from one frozen transcript so they stay disjoint even as earlier chunks
    /// tighten the live bounds. `1` is byte-identical to the historical single
    /// sweep. `NY_INTERM_ROW_CHUNKS` overrides for A/B.
    #[serde(default = "default_root_comprehensive_gpu_interm_chunks")]
    pub root_comprehensive_gpu_interm_chunks: usize,
    // --- Post-C-matrix survivor Patch-CROWN (#post-c-survivor) ---
    /// After the cheap sound multi-objective root C-matrix candidate, rerun at
    /// most 16 still-unverified rows through one resource-bounded generic full
    /// DAG Patch-CROWN backward and sound-intersect the result.
    ///
    /// This is an experimental, default-OFF root-only policy. The CLI resolves
    /// the exact `NY_ROOT_POST_C_SURVIVOR=1` opt-in once into this typed field;
    /// the solver never reads or mutates process-global environment state.
    #[serde(default)]
    pub root_post_c_survivor: bool,
    // --- β/α-ascent graft for the multi-objective dense-spec lane (#mo-beta-graft) ---
    /// Run the wide GPU segment-lane β/α ascent to OPTIMIZE the split
    /// multipliers, then EVALUATE through the tight dense-spec batched
    /// primitive with those multipliers folded in, taking the elementwise
    /// tightest of {dense-spec bound, ascended wide bound}.
    ///
    /// Motivation (metaroom 6cnn, measured): the wide segment lane's base
    /// relaxation is 2.4-2.9x LOOSER than the dense-spec primitive, so
    /// routing conv chains onto it wholesale (NY_BAB_CHAIN_WIDE=1) REPLACES a
    /// tight bound with a loose-ascended one; but the ascent's optimized β
    /// genuinely tightens the dense bound when folded back in.
    ///
    /// SOUND: both inputs are valid bounds on the same spec rows over the
    /// same subdomain (the ascended bound folds the SAME β entries — built by
    /// `with_constraint` from this domain's split history, values only moved
    /// under a β>=0 clamp — that the dense backward's `apply_beta_contribution`
    /// folds), so the per-row intersection encloses the true range.
    ///
    /// Default: false. Env override: `NY_MO_BETA_GRAFT=1` forces on,
    /// `NY_MO_BETA_GRAFT=0` forces off (A/B).
    #[serde(default)]
    pub mo_beta_graft: bool,
    /// Arm the multi-objective wave-batched kFSB selector by default (env
    /// NY_MO_KFSB overrides either way). Set true only for benchmarks measured
    /// Pareto (CIFAR-100).
    ///
    /// A/B evidence (cifar100, live run): 9/9 instances strictly better bound
    /// trajectory, 0 verdict regressions, wall never extended. Scoped to the
    /// CIFAR-100 presets only so every other lane (relational, ACAS Xu,
    /// relusplitter, …) stays byte-identical. Kill switch: `NY_MO_KFSB=0`
    /// force-off, `NY_MO_KFSB=1` force-on; unset falls back to this field.
    #[serde(default)]
    pub use_kfsb_multi_branching: bool,
    /// Reuse a strictly-authorized scalar lower certificate from the historical
    /// wave-kFSB child simulations when it proves a committed child complete.
    ///
    /// Default false keeps every unqualified lane on the advisory-only path.
    /// An absent `NY_MO_KFSB_CERT_REUSE` inherits this typed setting, literal
    /// `1` force-arms it, and every other present byte string force-disarms it
    /// (including malformed and non-Unicode values).
    #[serde(default)]
    pub kfsb_cert_reuse: bool,
    /// Use the aggregation-critical objective row for the full graph kFSB
    /// scorer on auto-selected, high-dimensional conjunctive workloads.
    ///
    /// This is a runtime routing decision rather than a reusable preset knob:
    /// the CLI sets it only when model-aware `auto` branching resolves to kFSB
    /// for the high-dimensional/large-ReLU class. The selector additionally
    /// requires conjunctive aggregation. Default false preserves the cheaper
    /// historical intercept ranking for every explicit-kFSB, MIP-fallback, and
    /// disjunctive lane.
    #[serde(default)]
    pub use_multi_objective_critical_kfsb: bool,
    // --- Multi-Depth ReLU Splitting (#2767) ---
    /// Max ReLU neurons to split per domain. Default: 1 (single-neuron).
    /// NY-only defensive expansion cap; it is not alpha-beta-CROWN's
    /// `max_split_depth` semantic.
    #[serde(default = "default_max_relu_split_depth")]
    pub max_relu_split_depth: usize,
    /// Queue fraction triggering multi-depth. Default: 0.5.
    /// Reference: alpha-beta-CROWN `min_batch_size_ratio`.
    #[serde(default = "default_min_batch_fill_ratio")]
    pub min_batch_fill_ratio: f32,
    // --- lA Warm-Start (GPU BaB) ---
    /// Enable lA warm-start in the GPU BaB backward pass.
    ///
    /// When enabled (default), child domains inherit cached linear bound
    /// coefficients (lA) from their parent's backward pass and use them to
    /// seed the next backward pass at the branch point instead of recomputing
    /// from the output node. This skips redundant backward computation for
    /// layers above the branch point.
    ///
    /// Disable with `--no-la-warm-start` for A/B benchmarking.
    ///
    /// # Reference
    /// alpha-beta-CROWN: `complete_verifier/tensor_storage.py` (all_lAs)
    /// Issue: #1564, #1669
    #[serde(default = "default_enable_la_warm_start")]
    pub enable_la_warm_start: bool,
    // --- IBP Enhancement for Input Split (#3813) ---
    /// Enable IBP-enhanced CROWN in the input-split BaB loop.
    ///
    /// When enabled, each domain (root and children) is first evaluated with IBP.
    /// Domains verified by IBP alone skip the expensive CROWN backward pass.
    /// Domains that remain undecided still run fresh spec-guided CROWN, but
    /// the current-domain IBP intermediates are threaded in as reference bounds
    /// so each nonlinear layer can keep the tighter of `{fresh CROWN, IBP}`.
    ///
    /// Reference name: alpha-beta-CROWN `bab.branching.input_split.ibp_enhancement`
    /// (`alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/input_split/bounding.py:131-136`)
    ///
    /// Default: false (no IBP enhancement).
    #[serde(default)]
    pub input_split_ibp_enhancement: bool,

    /// Admit the bounded proof-only nonnegative conic closure for the two-row
    /// conjunctive shape `[1, 0] <= +0 AND [0, -1] <= -0` during graph input
    /// splitting. Derived weighted rows are never added to the `VnnLibSpec`,
    /// attack paths, witness paths, or ReLU-split searches.
    ///
    /// Admission additionally requires verdict-only artifact authority because
    /// the external certificate format cannot currently encode the derived
    /// rows' provenance.
    ///
    /// Default: false (historical objective set and call path).
    #[serde(default)]
    pub input_split_conic_objective: bool,

    /// Maximum deferred-rebound tranche for an authenticated affine-conic
    /// reordered input-split search.
    ///
    /// Smaller tranches return newly bounded domains to the priority heap
    /// sooner, discard less completed work at a deadline, and bound the live
    /// `LinearBounds` matrices carried by the popped domain set. This setting
    /// has no effect outside the exact affine-conic route. Must be at least 1.
    #[serde(default = "default_input_split_conic_queue_refresh_batch_size")]
    pub input_split_conic_queue_refresh_batch_size: usize,

    /// Decompose the canonical exactly-two-singleton disjunction into
    /// sequential single-objective DomainList searches.
    ///
    /// Default false. This route is an explicit category treatment, not a
    /// shape-wide heuristic: the same AST occurs in unrelated cGAN and
    /// ml4acopf properties whose grouped search policy must remain unchanged.
    #[serde(default)]
    pub input_split_independent_singleton_disjunction: bool,

    // --- Domain-stacked batched rebound for Input Split (#cgan-batched-stack) ---
    /// Enable the domain-stacked dense-spec batched rebound in the input-split
    /// BaB loop.
    ///
    /// When enabled, the batched dense-spec CROWN backward kernel:
    /// 1. stacks the active domains' spec rows into ONE backward call per
    ///    Conv2d / ConvTranspose2d / BatchNorm node (instead of one call per
    ///    domain), amortizing the per-call im2col / weight-reshape / sound f64
    ///    recompute across the batch. Sound: those backward maps are
    ///    row-independent linear operators; any per-node box consumed by the
    ///    dispatch (BatchNorm certified-error discharge / precompute widening)
    ///    uses the elementwise HULL of the active domains' boxes, which can
    ///    only widen (equal-or-looser than the per-domain path, never tighter).
    ///    Any shape/op the stacked path cannot handle falls back to the
    ///    per-domain loop unchanged.
    /// 2. when `input_split_ibp_enhancement` is also set, seeds each domain's
    ///    intermediate-bounds cache with fresh per-subdomain IBP intersected
    ///    with the shared warmup reference bounds (the intersection of two
    ///    sound enclosures — sound, and matching the rayon per-domain
    ///    `ibp_enhancement` path that the batched kernel previously omitted per
    ///    #4210). Without this, all empty-history domains share bit-identical
    ///    backward relaxations and only the final concretization varies.
    ///
    /// Default: false (per-domain backward loop, verbatim shared cache — the
    /// historical behavior).
    #[serde(default)]
    pub input_split_stacked_rebound: bool,

    /// Run eligible per-domain warm-alpha refinements concurrently during the
    /// deferred reordered input-split rebound.
    ///
    /// This is a preset-scoped activation gate: the global
    /// `NY_INPUT_SPLIT_WARM_PARALLEL` environment setting cannot arm a category
    /// whose preset leaves this false. Default false preserves the historical
    /// serial refinement loop for every existing preset.
    #[serde(default)]
    pub input_split_warm_parallel: bool,

    /// Evaluate complete-clip child-local override rebounds two at a time in
    /// the scalar and dense-spec deferred reordered input-split loops.
    ///
    /// The executor is deliberately fixed at two workers: each worker owns a
    /// full CROWN carrier, so unconstrained Rayon fan-out can multiply peak
    /// memory by the host's core count. Results are collected and applied in
    /// domain order, preserving deterministic heap mutation and error choice.
    /// A conservative workload estimate must also fit within one eighth of a
    /// live, kernel-enforced process envelope; otherwise the route falls back
    /// to the serial reference. This limits incremental OOM risk but is not a
    /// general whole-CROWN peak-memory guarantee, so presets should opt in only
    /// for measured small models. This typed gate takes precedence over the
    /// separate collection-verify shortcut whenever the environment selector
    /// remains armed, including when memory admission selects serial execution.
    /// The environment kill switch restores independent shortcut selection. A
    /// preset must opt in before
    /// `NY_INPUT_SPLIT_OVERRIDE_PARALLEL` may select the parallel arm. Default
    /// false preserves the historical serial path.
    #[serde(default)]
    pub input_split_override_parallel: bool,

    /// Arm Saturation-Escape Branching (SEB, #nn4sys-seb-dark) by default for
    /// this config: the input-split dim scorer that de-saturates a binding
    /// sigmoid/tanh logit (`engine/graph/input_split/sat_escape.rs`), plus the
    /// disjunctive precheck-fraction cap that reserves per-clause BaB budget
    /// for the brancher (ny-cli `verify/disjunctive.rs`). Advisory + budget
    /// only — the split partition stays an exact cover and every verdict path
    /// is unchanged, so soundness is untouched either way.
    ///
    /// Probe (scratchpad/invprop_ab/mscn_dual_SEB_probe_RESULTS_2026-07-20.txt):
    /// 1-D mscn dual disjuncts close in 7–20 leaves, 2-D in 57–369 with SEB vs
    /// 697/timeout blind, while CROWN-per-box flips 0/1200. Preset opt-in
    /// (nn4sys); every preset that does not name the key keeps today's dark
    /// default. Env `NY_SAT_ESCAPE_BRANCH` overrides either way: `1` force-on,
    /// `0` kill switch, unset falls back to this field
    /// (see [`Self::sat_escape_branch_armed`]).
    #[serde(default)]
    pub sat_escape_branch: bool,

    /// INTERNAL companion to `input_split_stacked_rebound`: enables the fresh
    /// per-domain IBP intersect in the batched kernel's forward stage. Set by
    /// the input-split batched adapter (`stacked_rebound && ibp_enhancement`);
    /// not a preset surface. Default: false (verbatim shared-cache clone).
    #[serde(default)]
    pub input_split_batched_ibp_refresh: bool,

    /// #relational-bab lever 1 (default OFF): in the multi-clause DISJUNCTIVE
    /// input-split rebound, derive the spec-row bounds by interval arithmetic
    /// over the per-domain intermediate collection's OUTPUT entry and SKIP the
    /// spec backward for domains that already verify. For pure `±e_i` band
    /// rows with per-node CROWN-IBP intermediates the two are BIT-IDENTICAL
    /// (measured on the relational ACAS difference nets), so the backward is
    /// pure redundancy for the verified majority at the frontier; survivors
    /// still run the standard spec backward over the SAME collected
    /// intermediates (no recollection) so split scoring / clipping keep their
    /// linear bounds. Sound: the interval combination of a sound output
    /// enclosure is a sound spec bound; the monotonic parent-tighten guard is
    /// applied identically on both paths.
    #[serde(default)]
    pub input_split_collection_verify_shortcut: bool,

    /// #relational-bab lever 2 (default OFF): honor `input_split_depth` in
    /// the multi-clause DISJUNCTIVE input-split lane — split the top-k SB
    /// dims per pop (up to `2^k` children exactly covering the parent),
    /// mirroring the conjunctive multi-objective lane. OFF preserves the
    /// historical single-dim split byte-identically.
    #[serde(default)]
    pub input_split_disjunctive_multi_dim: bool,

    /// Offer every still-undecided grouped-disjunctive input subdomain to the
    /// attached input-box leaf oracle before the Graph-MIP edge escalation
    /// collects intermediate node bounds.
    ///
    /// The oracle receives the exact input box and complete grouped objective
    /// layout. Only a deadline-valid `VerifiedAllRows` may discharge the
    /// domain; `Undecided`, advisory violations, malformed requests, and late
    /// results fall through to the historical edge-MIP/split path unchanged.
    /// This foundation requires certification of every objective row; grouped
    /// one-row-per-clause authority is deferred until a typed certified-row
    /// mask can be validated by the caller.
    /// Default false keeps existing configurations and Graph-MIP-only oracles
    /// inert at this new seam.
    #[serde(default)]
    pub input_split_input_leaf_oracle: bool,

    /// #relational-bab EDGE-DOMAIN ESCALATION (default OFF): in the
    /// multi-clause DISJUNCTIVE input-split lane, a near-verified deep domain
    /// (every unverified single-row clause within `input_split_edge_milp_gap`
    /// of its threshold, `depth >= input_split_edge_milp_depth`) is offered to
    /// the attached [`GraphMipLeafOracle`](crate::beta_crown::graph_mip_leaf)
    /// with NO split premises (the input box IS the subdomain) before being
    /// split further. `VerifiedAllRows` (certified-UNSAT only, the leaf lane's
    /// 0-wrong contract) counts the domain verified; anything else requeues
    /// unchanged. This DECIDES relaxation-floor edge domains — boundary
    /// -unstable neuron pairs whose plain-CROWN slack no amount of input
    /// splitting eliminates — instead of splitting them forever. Inert unless
    /// an oracle is attached.
    #[serde(default)]
    pub input_split_edge_milp: bool,
    /// Edge gate: max distance-to-threshold (over unverified rows) for a
    /// domain to qualify as an escalation candidate.
    #[serde(default = "default_edge_milp_gap")]
    pub input_split_edge_milp_gap: f32,
    /// Edge gate: minimum BaB depth before escalation is considered.
    #[serde(default = "default_edge_milp_depth")]
    pub input_split_edge_milp_depth: usize,

    /// #relational-bab OPTION B (default OFF): α-OPTIMIZED slopes on EDGE
    /// domains. A popped near-verified domain (same gap/depth gates as the
    /// MILP escalation) gets a per-domain α-CROWN pass — optimized lower
    /// slopes over its exact sub-box — on its unverified rows BEFORE
    /// splitting; verified ⇒ done, still-short ⇒ split with the improved
    /// (monotonicity-guarded) bounds. Composes with the MILP consult (α
    /// first; the certified MILP finishes when the free-binary count
    /// collapses at depth). Measured α-over-plain-CROWN gains (1e-3..1e-1)
    /// cover most of the −0.0002..−0.03 relaxation floor.
    #[serde(default)]
    pub input_split_edge_alpha: bool,
    /// Per-WAVE cap on α edge passes (the pop order is worst-gap-first, so
    /// the cap keeps the most negative gaps; ~50-200ms per pass).
    #[serde(default = "default_edge_alpha_top")]
    pub input_split_edge_alpha_top: usize,
    /// α iterations per edge pass (modest; deadline-guarded).
    #[serde(default = "default_edge_alpha_iters")]
    pub input_split_edge_alpha_iters: usize,

    // --- Reorder BaB for Input Split (#3870) ---
    /// Use the reordered BaB loop for input splitting.
    ///
    /// In regular mode, the loop order is: pop → split → clip → bound children → enqueue.
    /// In reorder mode, the loop order is: pop → bound → filter verified → split → clip → enqueue.
    ///
    /// Reorder mode bounds the domain before deciding whether to split. Verified
    /// domains are filtered before splitting, saving clip + child construction
    /// overhead. Split decisions use fresh CROWN linear coefficients from the
    /// bounding step.
    ///
    /// Reference: alpha-beta-CROWN `bab.branching.input_split.reorder_bab`
    /// (`alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/input_split/batch_branch_and_bound.py:58-63`)
    ///
    /// Default: false (regular BaB loop order).
    #[serde(default)]
    pub reorder_bab: bool,

    /// Domain-count threshold for adversarial checking during input-split BaB.
    ///
    /// After `adv_check` domains have been explored, run a lightweight PGD probe
    /// on the worst queued domains each iteration to find counterexamples early.
    /// - `-1`: disabled (never run adv_check during BaB)
    /// - `0`: run from the first iteration
    /// - `N > 0`: run after N domains have been explored
    ///
    /// Reference: alpha-beta-CROWN `bab.branching.input_split.adv_check`
    /// (`alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/input_split/batch_branch_and_bound.py:512-522`)
    ///
    /// Default: -1 (disabled).
    #[serde(default = "default_adv_check")]
    pub adv_check: i32,

    // --- Adaptive Batch Sizing (#4303) ---
    /// Enable opt-in adaptive BaB microbatch sizing.
    ///
    /// By itself this retains the legacy full-batch enlargement policy. When
    /// the independent `NY_ADAPTIVE_MICROBATCH_CONTROLLER=1` gate is also set
    /// exactly, graph ReLU-split and DomainList input-split routes grow only
    /// when their tensor-byte estimate leaves the configured reserve available.
    /// They back off on retryable allocation/dispatch refusals and may shrink
    /// after underfill, memory pressure, or repeated long passes. Other BaB
    /// routes retain the legacy helper, capped at
    /// `AUTO_ENLARGE_BATCH_CAP` (8192).
    ///
    /// Reference: alpha-beta-CROWN `auto_LiRPA/utils.py:348-381` (`AutoBatchSize`).
    /// WGPU/Metal currently uses NY's configured/system-memory backend-budget
    /// fallback plus measured tensor bytes; it does not query live free VRAM.
    ///
    /// Default: false. The adaptive controller's independent environment gate
    /// is also default-dark; unset, `0`, or malformed values preserve the
    /// historical execution and fallback path even for existing presets that
    /// set this field.
    #[serde(default)]
    pub auto_enlarge_batch_size: bool,

    // --- #dd-zonotope per-category admission overrides (#metaroom-ddzono) ---
    /// Preset-driven override of `DdZonoConfig::min_input_numel`.
    ///
    /// These four knobs make the certified double-double zonotope's
    /// ADMISSION caps reachable from a category preset, because the scored
    /// `ny vnncomp v1` entry point sets no environment variables. They only
    /// resize the fail-closed detector's blast-radius/resource caps; every
    /// soundness gate — the self-policing precision gate, the rounding-channel
    /// safety factor, `dd_selfcheck`, the FP-environment probe, and outward
    /// f64→f32 narrowing at the verdict — is untouched and NOT preset-
    /// configurable. `None` (the default for every existing preset and
    /// serialized config) is byte-identical to today. An explicitly set
    /// `NY_DD_ZONOTOPE_*` environment variable keeps precedence over these
    /// (see `DdZonoConfig::with_admission_overrides`).
    #[serde(default)]
    pub dd_zonotope_min_input_numel: Option<usize>,
    /// Preset-driven override of `DdZonoConfig::max_k` (perturbed-input cap).
    #[serde(default)]
    pub dd_zonotope_max_k: Option<usize>,
    /// Preset-driven override of `DdZonoConfig::max_generators` (live
    /// generator-column cap; exceeding it still FAILS CLOSED).
    #[serde(default)]
    pub dd_zonotope_max_generators: Option<usize>,
    /// Preset-driven override of `DdZonoConfig::collect_interm`
    /// (`#dd-zono-interm`): populate the certified per-node enclosure map so
    /// the root coordinator may INTERSECT it into the stored intermediate
    /// bounds. Intersection of two certified enclosures of the same quantity
    /// is itself a certified enclosure, so this is bound-tightening only.
    #[serde(default)]
    pub dd_zonotope_collect_interm: Option<bool>,

    // --- Phase Budget Policy (#2206) ---
    /// Phase-level time budget fractions for β-CROWN verification.
    ///
    /// Centralizes the timeout policy that was previously scattered across
    /// `attack_budget.rs`, `sequential.rs`, `graph.rs`, `disjunctive.rs`,
    /// and `mod.rs` as hardcoded magic constants.
    ///
    /// Default values preserve today's behavior exactly.
    /// Design: `designs/2026-03-16-issue-2206-adaptive-phase-budgeting.md`
    #[serde(default)]
    pub phase_budget: PhaseBudgetConfig,
}

/// Maximum batch size for auto-enlargement (#4303).
/// Matches alpha-beta-CROWN's practical cap for GPU memory safety.
pub const AUTO_ENLARGE_BATCH_CAP: usize = 8192;

fn default_adv_check() -> i32 {
    -1
}

/// Pure resolver for the typed kFSB certificate-reuse policy. Kept separate so
/// tests can cover non-Unicode environment values without mutating process
/// state; production consumers use [`BetaCrownConfig::kfsb_cert_reuse_armed`].
pub(crate) fn kfsb_cert_reuse_from_raw(configured: bool, raw: Option<&std::ffi::OsStr>) -> bool {
    raw.map_or(configured, |value| value == std::ffi::OsStr::new("1"))
}

impl Default for BetaCrownConfig {
    fn default() -> Self {
        Self {
            verification_artifact_authority: VerificationArtifactAuthority::CertificateExport,
            max_domains: 100_000,
            max_queue_size: default_max_queue_size(),
            // #ml4acopf-bab-queue-mem: 0 = unlimited. Opt-in per preset so
            // every category that does not set it is byte-identical.
            max_queue_bytes: 0,
            timeout: Duration::from_mins(5),
            max_depth: 100,
            use_alpha_crown: true,
            use_forward_bounds: false,
            optimize_disjuncts_separately: false,
            root_alpha_cap_secs: None,
            root_alpha_phase_checkpoint: false,
            atomic_root_c_margin_iterations: 0,
            use_crown_ibp: false, // Disabled by default (more expensive)
            max_crown_ibp_nodes: None,
            crown_ibp_per_node_floor_secs: None,
            crown_ibp_per_node_cap_secs: None,
            crown_backward_layers: None,
            alpha_config: AlphaCrownConfig::default(),
            branching_heuristic: BranchingHeuristic::LargestBoundWidth,
            input_split_depth: default_input_split_depth(),
            input_split_coeff_thresh: default_input_split_coeff_thresh(),
            input_split_alpha_iteration: default_input_split_alpha_iteration(),
            input_split_lr_alpha: default_input_split_lr_alpha(),
            input_split_touch_zero_score: default_input_split_touch_zero_score(),
            input_split_sb_margin_weight: default_input_split_sb_margin_weight(),
            input_split_sb_sum: false,
            input_split_sb_primary_spec: None,
            fsb_candidates: default_fsb_candidates(),
            kfsb_reduce_op: KfsbReduceOp::default(), // Min (conservative)
            depth_two_branch_lookahead: DepthTwoBranchLookaheadConfig::default(),
            beta_lr: 0.05,      // α,β-CROWN default: 0.05
            beta_iterations: 0, // Per-domain iterations disabled by default for throughput
            beta_tolerance: 1e-5,
            early_stop_patience: default_early_stop_patience(),
            root_beta_iterations: default_root_beta_iterations(), // Root-level optimization
            beta_max_depth: default_beta_max_depth(), // Limit per-domain optimization depth
            use_analytical_beta_gradients: true, // Use analytical gradients for ~3x faster optimization
            alpha_lr: 0.01,                      // α,β-CROWN default: 0.01 (much lower than init!)
            alpha_momentum: true,                // Use momentum for α updates
            batch_size: 64,                      // Process 64 domains in parallel (GPU-optimized)
            build_batch_size: None,
            parallel_children: true, // Enable parallel child creation by default
            use_adaptive: false,     // Disabled by default for backward compatibility
            adaptive_config: AdaptiveOptConfig::default(),
            enable_cuts: false, // Disabled by default for backward compatibility
            conv_mode: ConvMode::default(), // Auto: patches unless cuts enabled
            max_cuts: default_max_cuts(),
            min_cut_depth: default_min_cut_depth(),
            enable_near_miss_cuts: false, // Disabled by default
            near_miss_margin: default_near_miss_margin(),
            enable_proactive_cuts: false, // Disabled by default
            max_proactive_cuts: default_max_proactive_cuts(),
            enable_biccos_constraint_strengthening: false, // Disabled by default
            biccos_drop_ratio: default_biccos_drop_ratio(),
            lambda_opt_interval: default_lambda_opt_interval(),
            lambda_lr: default_lambda_lr(),
            cut_eviction_policy: CutEvictionPolicy::default(),
            cut_stale_iters: default_cut_stale_iters(),
            cut_hard_stale_iters: default_cut_hard_stale_iters(),
            cut_lambda_min: default_cut_lambda_min(),
            cut_proactive_fraction: default_cut_proactive_fraction(),
            cut_score_weights: CutScoreWeights::default(),
            enable_biccos_cold_start: false, // BICCOS cold-start gating
            biccos_min_verified: default_biccos_min_verified(),
            biccos_min_verified_rate: default_biccos_min_verified_rate(),
            biccos_verified_rate_window: default_biccos_verified_rate_window(),
            biccos_min_cuts: default_biccos_min_cuts(),
            biccos_min_bound_gain: default_biccos_min_bound_gain(),
            biccos_bound_gain_window: default_biccos_bound_gain_window(),
            biccos_cold_max_iters: default_biccos_cold_max_iters(),
            biccos_cut_window: default_biccos_cut_window(),
            biccos_min_cut_yield: default_biccos_min_cut_yield(),
            biccos_cut_yield_window: default_biccos_cut_yield_window(),
            biccos_cut_yield_patience: default_biccos_cut_yield_patience(),
            verify_upper_bound: false,
            enable_pgd_attack: false,
            pgd_restarts: default_pgd_restarts(),
            pgd_steps: default_pgd_steps(),
            pgd_restart_when_stuck: false,
            pgd_initialization: PgdInitialization::Uniform,
            pgd_osi_steps: default_pgd_osi_steps(),
            pgd_gama: false,
            pgd_optimizer: PgdOptimizer::AdamClipping,
            pgd_alpha_mode: PgdAlphaMode::Auto,
            pgd_lr_decay: default_pgd_lr_decay(),
            pgd_surrogate_sign_gradient: false,
            pgd_dense_low_dim_sweep: false,
            pgd_dense_sweep_max_dims: default_pgd_dense_sweep_max_dims(),
            pgd_dense_sweep_points: default_pgd_dense_sweep_points(),
            enable_relaxed_clip: false,
            relaxed_clip_iterations: default_relaxed_clip_iterations(),
            input_clip_type: InputClipType::default(),
            input_split_fresh_domain_clip: false,
            clip_neuron_selection_ratio: default_clip_neuron_selection_ratio(),
            enable_clip_interm_domain: false,
            clip_interm_topk: default_clip_interm_topk(),
            clip_in_alpha_crown: false,
            clip_interm_prune: false,
            clip_interm_use_final_layer: false,
            enable_interm_transfer: true,
            root_interm_cuda_factory: false,
            mo_cuda_factory_engine_handoff: false,
            mo_cuda_bounded_shared_executor: false,
            root_crown_interm_dense_head: false,
            root_crown_interm_max_secs: default_root_crown_interm_max_secs(),
            root_crown_interm_max_dim: default_root_crown_interm_max_dim(),
            root_sparse_interm_crown: false,
            root_sparse_interm_crown_max_secs: default_root_sparse_interm_crown_max_secs(),
            root_sparse_interm_crown_max_dim: default_root_sparse_interm_crown_max_dim(),
            root_sparse_interm_crown_max_rows: default_root_sparse_interm_crown_max_rows(),
            root_sparse_interm_crown_max_targets: default_root_sparse_interm_crown_max_targets(),
            root_comprehensive_gpu_interm: false,
            // #bab-floor: None, not 0.0 — absent means the arbitration does
            // not exist, which is the byte-identical shipped ladder. 0.0 is a
            // different statement: an explicit kill switch that still leaves
            // the two shares parseable.
            root_bab_reserve_frac: None,
            root_spec_frac: None,
            root_alpha_frac: None,
            root_comprehensive_gpu_interm_chunks: default_root_comprehensive_gpu_interm_chunks(),
            root_post_c_survivor: false,
            mo_beta_graft: false, // #mo-beta-graft (env NY_MO_BETA_GRAFT overrides)
            use_kfsb_multi_branching: false, // #kfsb-multi: CIFAR-100 preset opt-in (env NY_MO_KFSB overrides)
            kfsb_cert_reuse: false, // proof-carrying kFSB simulation reuse; preset-scoped opt-in
            use_multi_objective_critical_kfsb: false, // auto high-dimensional conjunction only
            max_relu_split_depth: default_max_relu_split_depth(),
            min_batch_fill_ratio: default_min_batch_fill_ratio(),
            enable_la_warm_start: default_enable_la_warm_start(),
            input_split_ibp_enhancement: false,
            input_split_conic_objective: false,
            input_split_conic_queue_refresh_batch_size:
                default_input_split_conic_queue_refresh_batch_size(),
            input_split_independent_singleton_disjunction: false,
            input_split_stacked_rebound: false, // #cgan-batched-stack
            input_split_warm_parallel: false,   // #cgan-warm-par (preset-scoped opt-in)
            input_split_override_parallel: false, // bounded complete-clip rebound fan-out
            sat_escape_branch: false, // #nn4sys-seb-dark: nn4sys preset opt-in (env NY_SAT_ESCAPE_BRANCH overrides)
            input_split_batched_ibp_refresh: false, // #cgan-batched-stack
            input_split_collection_verify_shortcut: false, // #relational-bab lever 1
            input_split_disjunctive_multi_dim: false, // #relational-bab lever 2
            input_split_input_leaf_oracle: false, // input-box-only certified leaf seam
            input_split_edge_milp: false, // #relational-bab edge escalation
            input_split_edge_milp_gap: default_edge_milp_gap(),
            input_split_edge_milp_depth: default_edge_milp_depth(),
            input_split_edge_alpha: false, // #relational-bab option B
            input_split_edge_alpha_top: default_edge_alpha_top(),
            input_split_edge_alpha_iters: default_edge_alpha_iters(),
            reorder_bab: false,
            adv_check: -1,                              // #3870
            auto_enlarge_batch_size: false,             // #4303
            dd_zonotope_min_input_numel: None,          // #metaroom-ddzono (preset-scoped)
            dd_zonotope_max_k: None,                    // #metaroom-ddzono (preset-scoped)
            dd_zonotope_max_generators: None,           // #metaroom-ddzono (preset-scoped)
            dd_zonotope_collect_interm: None,           // #metaroom-ddzono (preset-scoped)
            phase_budget: PhaseBudgetConfig::default(), // #2206
        }
    }
}

impl BetaCrownConfig {
    /// Whether derived conjunctive input-split objectives have runtime verdict
    /// authority under this resolved configuration.
    ///
    /// This intentionally excludes property-shape checks, which remain at the
    /// graph frontend seam. Certificate-export runs decline until the external
    /// format can represent and replay the typed conic provenance.
    #[inline]
    pub fn input_split_conic_objective_eligible(&self) -> bool {
        self.input_split_conic_objective
            && self.verification_artifact_authority == VerificationArtifactAuthority::VerdictOnly
    }

    /// Resolve the proof-carrying kFSB simulation-reuse policy from its typed
    /// setting and exact environment override.
    ///
    /// This is the single production resolution point used by both the engine
    /// and effective-treatment reporting. Absence inherits the typed setting;
    /// literal `1` enables; every other present byte string disables.
    pub fn kfsb_cert_reuse_armed(&self) -> bool {
        kfsb_cert_reuse_from_raw(
            self.kfsb_cert_reuse,
            std::env::var_os("NY_MO_KFSB_CERT_REUSE").as_deref(),
        )
    }

    /// Whether cutting planes may influence certificate-bearing bounds.
    ///
    /// Cut proof authority is quarantined, and every consumer this gate used to
    /// guard has since been DELETED outright: the legacy post-hoc scalar
    /// contribution (applied after CROWN concretization instead of being folded
    /// through the backward relaxation), the `NY_CUT_FOLD` C2 registry fold, and
    /// the `arelu_cut` backward integration (the fold-through-the-relaxation
    /// variant). No bound-producing path reads the cut pool any more; the
    /// near-miss/proactive generators still create constraints without a proof.
    /// Keep this fail-closed gate separate from `enable_cuts` so even an
    /// internal caller that skips `validate()` cannot grant proof authority to
    /// the remaining cut machinery.
    ///
    /// Do NOT re-open this without a proof-producing, outward-rounded fold.
    #[inline]
    #[allow(dead_code)] // Deliberate quarantine seam, pinned directly by tests.
    pub(crate) fn cut_proof_authority_enabled(&self) -> bool {
        false
    }

    /// Whether CROWN backward should use Patches mode based on conv_mode and cuts.
    /// Reference: alpha-beta-CROWN `abcrown.py:228-231` — matrix when cuts enabled.
    #[inline]
    pub fn use_patches(&self) -> bool {
        self.conv_mode.use_patches(self.enable_cuts)
    }

    /// Resolve the Saturation-Escape Branching gate (#nn4sys-seb-dark): env
    /// `NY_SAT_ESCAPE_BRANCH=1` force-arms, `=0` force-disarms (the A/B kill
    /// switch), anything else falls back to [`Self::sat_escape_branch`] — the
    /// typed preset opt-in. Mirrors the `NY_MO_KFSB` /
    /// `use_kfsb_multi_branching` convention, and is the ONE resolution point
    /// for both consumers (the input-split SEB scorer gate and the disjunctive
    /// precheck budget cap), so the brancher and its budget can never disagree.
    pub fn sat_escape_branch_armed(&self) -> bool {
        match std::env::var("NY_SAT_ESCAPE_BRANCH").ok().as_deref() {
            Some("1") => true,
            Some("0") => false,
            _ => self.sat_escape_branch,
        }
    }

    /// Per-node CROWN-IBP time-budget overrides as the collector-facing type
    /// (#4413, #cgan-bn11-budget). All-`None` keeps the 2-second floor and
    /// selects the adaptive remaining-budget cap.
    #[inline]
    pub fn crown_ibp_per_node_time_budget(&self) -> crate::types::CrownIbpPerNodeTimeBudget {
        crate::types::CrownIbpPerNodeTimeBudget {
            floor_secs: self.crown_ibp_per_node_floor_secs,
            cap_secs: self.crown_ibp_per_node_cap_secs,
        }
    }

    /// ACAS-Xu optimized configuration preset.
    ///
    /// Configures input splitting with relaxed clipping for the ACAS-Xu benchmark.
    /// These settings match the α,β-CROWN VNN-COMP configuration for ACAS-Xu.
    ///
    /// Key settings:
    /// - Input splitting (not neuron branching)
    /// - Relaxed clipping enabled for tighter bounds
    /// - PGD attack for counterexample finding
    /// - Large batch size for GPU throughput
    ///
    /// Reference: VNN-COMP 2024 winning configuration.
    ///
    /// Disables alpha-CROWN: the reference uses plain CROWN (`bound_prop_method:
    /// crown`) for ACAS-Xu. With frozen root alpha reuse (#3453), alpha-CROWN
    /// intermediate bounds are computed once on the full root domain and reused
    /// for all sub-domains. For ACAS-Xu's tight 5-dim input space, these root
    /// bounds are too loose to help sub-domains, producing 0% verification.
    /// Plain CROWN uses per-domain IBP intermediates that tighten naturally as
    /// domains split, matching the reference's behavior.
    ///
    /// Reference: alpha-beta-CROWN exp_configs/vnncomp21/acasxu.yaml:7
    ///   `bound_prop_method: crown` (no alpha optimization)
    pub fn acas_xu() -> Self {
        Self {
            branching_heuristic: BranchingHeuristic::InputSplit,
            // Plain CROWN, not alpha-CROWN. Reference uses `bound_prop_method: crown`.
            // Frozen root alpha (#3453) regresses ACAS-Xu from 99.8% to 0% verified
            // because root intermediate bounds don't tighten for sub-domains.
            use_alpha_crown: false,
            use_forward_bounds: false,
            enable_relaxed_clip: true,
            enable_pgd_attack: true,
            pgd_restarts: 10_000,
            pgd_restart_when_stuck: true, // Reference: acasxu.yaml:20
            batch_size: 16_384,
            reorder_bab: true, // Reference: acasxu.yaml:13
            // Multi-dimensional input splitting: split the top 2 SB dims per parent,
            // producing up to 2^2=4 children that exactly cover the parent (BaB
            // completeness preserved). Mirrors alpha-beta-CROWN's storage_depth
            // batch-filling, but tuned empirically: on acasxu prop_1 (1_1/1_2/1_3),
            // depth=2 verifies in 29/105/85 domains (vs 32/234/184 at depth=1),
            // while depth>=3 regresses and depth=5 times out — the per-child screen
            // loop over-splits at high depth instead of cheaply batch-filling.
            input_split_depth: 2,
            ..Default::default()
        }
    }

    /// Build a runtime PGD config from the shared verifier settings.
    pub fn pgd_attack_config(
        &self,
        num_restarts: usize,
        num_steps: usize,
        deadline: Option<Instant>,
    ) -> PgdConfig {
        PgdConfig {
            num_restarts,
            num_steps,
            step_size: 0.01,
            spsa_delta: 0.001,
            seed: 42,
            parallel: true,
            deadline,
            restart_when_stuck: self.pgd_restart_when_stuck,
            initialization: self.pgd_initialization,
            osi_steps: self.pgd_osi_steps,
            gama_lambda: self.pgd_gama.then_some(GAMA_LAMBDA_DEFAULT),
            optimizer: self.pgd_optimizer,
            alpha_mode: self.pgd_alpha_mode,
            adam: AdamClippingParams {
                lr_decay: self.pgd_lr_decay,
                ..AdamClippingParams::default()
            },
            surrogate_sign_gradient: self.pgd_surrogate_sign_gradient,
            dense_low_dim_sweep: self.pgd_dense_low_dim_sweep,
            dense_sweep_max_dims: self.pgd_dense_sweep_max_dims,
            dense_sweep_points: self.pgd_dense_sweep_points,
        }
    }

    /// Returns true when the domain bounds are sufficient to prove the property.
    #[inline]
    pub(crate) fn domain_is_verified(&self, lower: f32, upper: f32, threshold: f32) -> bool {
        Self::domain_is_verified_for_mode(self.verify_upper_bound, lower, upper, threshold)
    }

    /// Returns true when the domain bounds are sufficient to prove violation.
    #[inline]
    pub(crate) fn domain_is_violation(&self, lower: f32, upper: f32, threshold: f32) -> bool {
        Self::domain_is_violation_for_mode(self.verify_upper_bound, lower, upper, threshold)
    }

    /// Mode-parametrized verification check used by shared dispatch/check helpers.
    ///
    /// Returns `false` when any input is non-finite (NaN or Inf). NaN bounds
    /// must never produce a Verified result. Inf bounds indicate propagation
    /// failure (e.g., reciprocal zero-crossing), not genuine verification.
    /// Inf threshold would trivially verify/reject all domains (#2993).
    /// A single `is_finite()` check covers both NaN and Inf per IEEE 754.
    #[inline]
    pub(crate) fn domain_is_verified_for_mode(
        verify_upper_bound: bool,
        lower: f32,
        upper: f32,
        threshold: f32,
    ) -> bool {
        // Reject all non-finite inputs: NaN (propagation corruption) and Inf
        // (propagation failure, e.g., reciprocal zero-crossing). Inf threshold
        // would make every domain trivially verified or never verified (#2993).
        if !lower.is_finite() || !upper.is_finite() || !threshold.is_finite() || lower > upper {
            return false;
        }
        if verify_upper_bound {
            upper < threshold
        } else {
            lower > threshold
        }
    }

    /// Mode-parametrized violation check used by shared dispatch/check helpers.
    ///
    /// Returns `false` when any input is non-finite (NaN or Inf). NaN bounds
    /// must never produce a Violation result. Inf bounds indicate propagation
    /// failure, not a genuine violation. Inf threshold would make every domain
    /// trivially violated or never violated (#2993).
    ///
    /// Also returns `false` for an INVERTED interval (`lower > upper`, #violdrop).
    /// This predicate is a PROOF — the decision-relevant bound is being asserted
    /// to hold over the whole region — and an inverted interval is numerically
    /// contradictory: a valid upper bound can never sit below a valid lower
    /// bound, so at most one of the two is sound and neither can be trusted as
    /// the proof side. Inversions DO occur here in practice (see
    /// `multi_objective/shared.rs::tighten_child_bounds_with_parent`, which detects and
    /// tolerates them, and the α-CROWN elementwise best-bound merge, which
    /// repairs them via `repair_inverted_bounds_nd`). Rejecting them makes this
    /// predicate strictly harder to satisfy: fewer drops, more search, never a
    /// verdict the previous predicate would not also have allowed.
    #[inline]
    pub(crate) fn domain_is_violation_for_mode(
        verify_upper_bound: bool,
        lower: f32,
        upper: f32,
        threshold: f32,
    ) -> bool {
        // Reject all non-finite inputs (#2993).
        if !lower.is_finite() || !upper.is_finite() || !threshold.is_finite() {
            return false;
        }
        // Reject inverted intervals (#violdrop).
        if lower > upper {
            return false;
        }
        if verify_upper_bound {
            lower >= threshold
        } else {
            upper < threshold
        }
    }

    /// Returns BaB queue priority for a domain with given bounds.
    ///
    /// Higher priority = process first. When `verify_upper_bound` is true, we want
    /// to process domains with the highest upper bound first (worst case). When false,
    /// we negate the lower bound so the max-heap orders by lowest lower bound first.
    #[inline]
    pub(crate) fn domain_priority(&self, lower: f32, upper: f32) -> Result<f32> {
        Self::domain_priority_for_mode(self.verify_upper_bound, lower, upper)
    }

    /// Mode-parametrized BaB queue priority.
    ///
    /// Returns `Err(NumericalInstability)` if either bound is non-finite, preventing
    /// zombie domains from entering the BaB queue (#2982, #3125).
    #[inline]
    pub(crate) fn domain_priority_for_mode(
        verify_upper_bound: bool,
        lower: f32,
        upper: f32,
    ) -> Result<f32> {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "BaB domain bounds are non-finite: lower={lower}, upper={upper}"
            )));
        }
        if lower > upper {
            return Err(NyError::NumericalInstability(format!(
                "BaB domain bounds are inverted: lower={lower}, upper={upper}"
            )));
        }
        Ok(if verify_upper_bound { upper } else { -lower })
    }

    /// Returns the relevant bound for threshold comparison.
    ///
    /// When `verify_upper_bound` is true, the upper bound is the decision-relevant
    /// bound (checking upper < threshold). When false, the lower bound is relevant
    /// (checking lower > threshold).
    #[inline]
    pub(crate) fn relevant_bound(&self, lower: f32, upper: f32) -> f32 {
        if self.verify_upper_bound {
            upper
        } else {
            lower
        }
    }

    /// Compute tightening gain from parent to child domain.
    ///
    /// Measures how much the relevant bound improved (moved toward threshold).
    #[inline]
    pub(crate) fn bound_gain(
        &self,
        parent_lower: f32,
        parent_upper: f32,
        child_lower: f32,
        child_upper: f32,
    ) -> f32 {
        // NaN-safe: propagate NaN instead of silently reporting 0.0 gain (#2643)
        if self.verify_upper_bound {
            nan_propagating_max(parent_upper - child_upper, 0.0)
        } else {
            nan_propagating_max(child_lower - parent_lower, 0.0)
        }
    }

    /// Human-readable verification direction for logging.
    ///
    /// Returns `"upper < threshold"` when verifying upper bound,
    /// `"lower > threshold"` when verifying lower bound.
    #[inline]
    pub(crate) fn verification_direction_str(&self) -> &'static str {
        if self.verify_upper_bound {
            "upper < threshold"
        } else {
            "lower > threshold"
        }
    }

    /// Counterexample direction symbol for logging.
    ///
    /// Returns `">="` when `verify_upper_bound` (counterexample is output >= threshold),
    /// `"<="` otherwise (counterexample is output <= threshold).
    #[inline]
    pub(crate) fn violation_direction_str(&self) -> &'static str {
        if self.verify_upper_bound {
            ">="
        } else {
            "<="
        }
    }

    /// Compatibility alias for BaB queue priority used by legacy call sites.
    ///
    /// #2682 requires queue ordering to respect the active verification mode:
    /// - `verify_upper_bound=true`: prioritize larger upper bounds
    /// - `verify_upper_bound=false`: prioritize smaller lower bounds
    ///
    /// This now matches [`domain_priority`] exactly so sequential and graph BaB
    /// paths cannot diverge due to helper selection.
    #[inline]
    pub(crate) fn violation_priority(&self, lower: f32, upper: f32) -> Result<f32> {
        Self::domain_priority_for_mode(self.verify_upper_bound, lower, upper)
    }

    /// Extract the worst-case FSB score from two child branches.
    ///
    /// For each branch, extracts the relevant bound (upper when `verify_upper_bound`,
    /// lower otherwise), then returns the negated worst (max upper / min lower).
    /// Returns `None` if both branches are `None`.
    #[inline]
    pub(crate) fn fsb_worst_case_score(
        &self,
        active: Option<(f32, f32)>,
        inactive: Option<(f32, f32)>,
    ) -> Option<f32> {
        if self.verify_upper_bound {
            let mut worst_upper = f32::NEG_INFINITY;
            if let Some((_, u)) = active {
                worst_upper = worst_upper.max(u);
            }
            if let Some((_, u)) = inactive {
                worst_upper = worst_upper.max(u);
            }
            if worst_upper == f32::NEG_INFINITY {
                None
            } else {
                Some(-worst_upper)
            }
        } else {
            let mut worst_lower = f32::INFINITY;
            if let Some((l, _)) = active {
                worst_lower = worst_lower.min(l);
            }
            if let Some((l, _)) = inactive {
                worst_lower = worst_lower.min(l);
            }
            if worst_lower == f32::INFINITY {
                None
            } else {
                Some(worst_lower)
            }
        }
    }

    /// Extract the relevant bound value from child domain bounds for kFSB scoring.
    ///
    /// Returns `-upper` when `verify_upper_bound`, `lower` otherwise.
    /// Returns `f32::NEG_INFINITY` when bounds are `None`.
    #[inline]
    pub(crate) fn child_bound_value(&self, bounds: Option<(f32, f32)>) -> f32 {
        match bounds {
            Some((lower, upper)) => {
                if self.verify_upper_bound {
                    -upper
                } else {
                    lower
                }
            }
            None => f32::NEG_INFINITY,
        }
    }

    /// Validate all learning rate and optimizer configuration fields.
    ///
    /// Checks top-level `beta_lr`, `alpha_lr`, and delegates to
    /// `adaptive_config.validate()` for Adam hyperparameters.
    /// Call after CLI flag application and preset merging.
    pub fn validate(&self) -> Result<()> {
        if Instant::now().checked_add(self.timeout).is_none() {
            return Err(NyError::InvalidConfig(format!(
                "timeout {:?} is too large for the platform monotonic clock",
                self.timeout
            )));
        }
        if self.enable_cuts || self.enable_near_miss_cuts || self.enable_proactive_cuts {
            return Err(NyError::InvalidConfig(
                "cut proof authority is quarantined: the legacy post-hoc scalar \
                 contribution is not a certified GCP-CROWN fold, and near-miss/proactive \
                 cuts are not proof-derived"
                    .to_string(),
            ));
        }
        if self.use_alpha_crown && self.use_forward_bounds {
            return Err(NyError::InvalidConfig(
                "use_forward_bounds cannot be enabled together with use_alpha_crown".to_string(),
            ));
        }
        if !self.beta_lr.is_finite() || self.beta_lr < 0.0 {
            return Err(NyError::InvalidConfig(format!(
                "beta_lr must be finite and >= 0, got {}",
                self.beta_lr
            )));
        }
        if !self.alpha_lr.is_finite() || self.alpha_lr < 0.0 {
            return Err(NyError::InvalidConfig(format!(
                "alpha_lr must be finite and >= 0, got {}",
                self.alpha_lr
            )));
        }
        if self.build_batch_size == Some(0) {
            return Err(NyError::InvalidConfig(
                "build_batch_size must be >= 1 when set".to_string(),
            ));
        }
        if self.input_split_conic_queue_refresh_batch_size == 0 {
            return Err(NyError::InvalidConfig(
                "input_split_conic_queue_refresh_batch_size must be >= 1".to_string(),
            ));
        }
        if self.atomic_root_c_margin_iterations > ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS {
            return Err(NyError::InvalidConfig(format!(
                "atomic_root_c_margin_iterations must be <= {}, got {}",
                ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS, self.atomic_root_c_margin_iterations
            )));
        }
        if self.input_split_depth == 0
            && matches!(self.branching_heuristic, BranchingHeuristic::InputSplit)
        {
            // Depth 0 selects no split dimensions, so every domain is
            // unsplittable and input-split BaB can make no progress.
            return Err(NyError::InvalidConfig(
                "input_split_depth must be >= 1 with the InputSplit branching heuristic"
                    .to_string(),
            ));
        }
        if self.enable_clip_interm_domain && self.clip_interm_prune {
            return Err(NyError::InvalidConfig(
                "clip_interm_prune is quarantined for certificate-backed Complete Clip; \
                 pruning authority is not implemented"
                    .to_string(),
            ));
        }
        if self.input_split_fresh_domain_clip
            && (!matches!(self.branching_heuristic, BranchingHeuristic::InputSplit)
                || !self.reorder_bab
                || !self.input_split_ibp_enhancement
                || !self.enable_relaxed_clip
                || !matches!(self.input_clip_type, InputClipType::Relaxed)
                || self.relaxed_clip_iterations == 0)
        {
            return Err(NyError::InvalidConfig(
                "input_split_fresh_domain_clip requires InputSplit branching, reorder_bab, \
                 input_split_ibp_enhancement, enabled relaxed clipping, relaxed clip type, \
                 and at least one relaxed clip iteration"
                    .to_string(),
            ));
        }
        self.phase_budget.validate()?;
        self.depth_two_branch_lookahead.validate()?;
        self.adaptive_config.validate()
    }
}

/// Hard resource cap for exact full-`C` root margin iterations.
pub const ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS: usize = 8;

/// Hard resource bounds for the July-2026 depth-2 lookahead experiment.
pub const DEPTH_TWO_LOOKAHEAD_MAX_CANDIDATES: usize = 15;
pub const DEPTH_TWO_LOOKAHEAD_MAX_ROUNDS: usize = 5;

/// Authority granted to branch-specific depth-2 lookahead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DepthTwoBranchLookaheadMode {
    /// Preserve the historical one-step kFSB path byte-for-byte.
    #[default]
    Off,
    /// Compute and report complete advice, but retain the one-step winner.
    Shadow,
    /// Allow complete, revalidated advice to choose a first-level root split.
    Select,
}

/// Typed policy for the bounded July-2026 Lookahead Branching experiment.
///
/// Depth is intentionally fixed at two. The defaults reproduce the published
/// alpha-beta-CROWN setting (15 BaBSR candidates, first five BaB rounds,
/// discount λ=0.5) while leaving the experiment disabled. Unknown serialized
/// keys are rejected deliberately: silently accepting a misspelled authority
/// or resource bound would make a sealed experiment's treatment ambiguous.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DepthTwoBranchLookaheadConfig {
    pub mode: DepthTwoBranchLookaheadMode,
    pub candidates: usize,
    pub top_rounds: usize,
    pub discount: f64,
}

impl Default for DepthTwoBranchLookaheadConfig {
    fn default() -> Self {
        Self {
            mode: DepthTwoBranchLookaheadMode::Off,
            candidates: DEPTH_TWO_LOOKAHEAD_MAX_CANDIDATES,
            top_rounds: DEPTH_TWO_LOOKAHEAD_MAX_ROUNDS,
            discount: 0.5,
        }
    }
}

impl DepthTwoBranchLookaheadConfig {
    /// Whether the typed experiment may run on this canonical outer BaB wave.
    #[inline]
    pub fn enabled_at_round(self, bab_round: usize) -> bool {
        self.mode != DepthTwoBranchLookaheadMode::Off && bab_round < self.top_rounds
    }

    fn validate(self) -> Result<()> {
        if !(1..=DEPTH_TWO_LOOKAHEAD_MAX_CANDIDATES).contains(&self.candidates) {
            return Err(NyError::InvalidConfig(format!(
                "depth_two_branch_lookahead.candidates must be in 1..={}, got {}",
                DEPTH_TWO_LOOKAHEAD_MAX_CANDIDATES, self.candidates
            )));
        }
        if !(1..=DEPTH_TWO_LOOKAHEAD_MAX_ROUNDS).contains(&self.top_rounds) {
            return Err(NyError::InvalidConfig(format!(
                "depth_two_branch_lookahead.top_rounds must be in 1..={}, got {}",
                DEPTH_TWO_LOOKAHEAD_MAX_ROUNDS, self.top_rounds
            )));
        }
        if !self.discount.is_finite() || !(0.0..=1.0).contains(&self.discount) {
            return Err(NyError::InvalidConfig(format!(
                "depth_two_branch_lookahead.discount must be finite and in [0, 1], got {}",
                self.discount
            )));
        }
        Ok(())
    }
}

/// Reduce operation for combining branch scores in kFSB.
/// Per alpha-beta-CROWN: min=conservative, max=optimistic, mean=balanced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum KfsbReduceOp {
    /// Conservative: take worst-case of both branches (default).
    #[default]
    Min,
    /// Optimistic: take best-case of both branches.
    Max,
    /// Balanced: average of both branches.
    Mean,
}

/// Input-domain clipping algorithm type.
///
/// Controls how input bounds are tightened using CROWN linear constraints
/// during input-split branch-and-bound.
///
/// - `Relaxed`: Closed-form 1D updates per dimension (fast, axis-aligned).
/// - `Complete`: LP-optimal tightening via Lagrangian dual coordinate ascent.
///   Tighter than relaxed because it accounts for cross-constraint dependencies.
///
/// Reference: alpha-beta-CROWN `clip_input_domain.clip_type`
/// (`abcrown_all_params.yaml:190`)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InputClipType {
    /// Closed-form relaxed clipping (matches alpha-beta-CROWN `clip_type: relaxed`).
    #[default]
    Relaxed,
    /// LP-optimal complete clipping via Lagrangian dual (matches `clip_type: complete`).
    Complete,
}

/// Convolution backward mode for CROWN propagation.
///
/// Controls whether Conv2d backward passes use Patches (per-position receptive
/// field) or Matrix (dense A-coefficient) representation.
///
/// Reference: alpha-beta-CROWN `general.conv_mode` (`abcrown.py:228-231`):
/// the reference forces Matrix mode when cuts are enabled because cutting
/// planes operate on flattened neurons and require dense A-matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ConvMode {
    /// Auto-select: Patches when cuts are disabled, Matrix when cuts are enabled.
    /// Matches reference `abcrown.py:228-231` policy.
    #[default]
    Auto,
    /// Always use Patches mode (per-position receptive field composition).
    Patches,
    /// Always use Matrix mode (dense A-coefficient backward).
    Matrix,
}

impl ConvMode {
    /// Returns `true` if the CROWN backward should use Patches mode.
    ///
    /// - `Patches` → always true
    /// - `Matrix` → always false
    /// - `Auto` → true unless cuts are enabled (reference: `abcrown.py:228-231`)
    pub fn use_patches(self, cuts_enabled: bool) -> bool {
        match self {
            Self::Patches => true,
            Self::Matrix => false,
            Self::Auto => !cuts_enabled,
        }
    }
}
