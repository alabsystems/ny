// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! VNN-COMP benchmark preset configuration loader.
//! Loads per-benchmark YAML presets that override `BetaCrownConfig` defaults.
//! CLI flags take precedence over preset values.

use anyhow::{Context, Result};
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
mod apply;
mod branching;
mod value_parse;
pub(crate) use apply::{apply_preset, resolve_use_alpha_from_bound_prop_method};
pub(crate) use branching::resolve_branching;

#[cfg(test)]
mod alpha_preset_tests;
#[cfg(test)]
mod attack_mode_tests;
#[cfg(test)]
mod bound_prop_mode_tests;
#[cfg(test)]
mod conv_mode_tests;
#[cfg(test)]
mod gpu_bab_sidecar_tests;
#[cfg(test)]
mod linearizenn_2024_tests;
#[cfg(test)]
mod phase_budget_tests;
#[cfg(test)]
mod preset_resolution_pin_tests;
#[cfg(test)]
mod relusplitter_biasfield_input_split_tests;
#[cfg(test)]
mod relusplitter_rsplitter_matrix_input_split_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod vnncomp_preset_tests;

/// Top-level preset configuration file structure.
///
/// Mirrors alpha-beta-CROWN's YAML config format for compatibility.
/// Supports both alpha-beta-CROWN's structure (solver: for batch_size, bab: for branching)
/// and ny's simplified structure (all under bab:).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct PresetConfig {
    /// General settings (root_path, csv_name, device).
    #[serde(default)]
    pub(crate) general: GeneralPreset,

    /// Model-loading options (alpha-beta-CROWN compatibility).
    #[serde(default)]
    pub(crate) model: ModelPreset,

    /// Attack/counterexample settings.
    #[serde(default)]
    pub(crate) attack: AttackPreset,

    /// Solver configuration (alpha-beta-CROWN uses this for batch_size, alpha/beta-crown settings).
    /// These values are merged into bab during apply_preset.
    #[serde(default)]
    pub(crate) solver: SolverPreset,

    /// Branch-and-bound configuration (core BetaCrownConfig overrides).
    #[serde(default)]
    pub(crate) bab: BabPreset,

    /// Margin-row twin-wall lane settings (#twinwall / #epoch-bab).
    #[serde(default)]
    pub(crate) margin_row: MarginRowPreset,
}

/// Margin-row lane budget policy, per benchmark category (#epoch-bab).
///
/// The lane runs AFTER the internal verifier returns unknown/timeout, so by
/// default it lives on leftover budget only and is strictly additive. A
/// category may opt into a RESERVE — seconds held back from the internal
/// verifier — but only where the measured production solve-time
/// distribution shows that tail is genuinely unused, because on a category
/// whose solves crowd the budget wall a reserve trades away real points
/// (measured: a 45 s reserve would forfeit 28 `sat_relu` and 10
/// `cifar100_2024` solves). Default 0 = no reserve.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct MarginRowPreset {
    /// Seconds reserved from the internal verifier for the lane.
    pub(crate) reserve_secs: Option<u64>,

    /// Release the reserve only for the sealed exact open-row allowlist and
    /// route those rows to the internal alpha/beta verifier. Unknown rows keep
    /// the configured reserve. Default `None`/`false` preserves the historical
    /// fixed-reserve policy.
    pub(crate) adaptive_reserve: Option<bool>,
}

/// Model-loading configuration (alpha-beta-CROWN compatibility).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct ModelPreset {
    /// ONNX conversion-time optimization flags.
    ///
    /// Accepts either a single string (`merge_linear`) or a YAML sequence.
    #[serde(default, deserialize_with = "string_or_vec_string")]
    pub(crate) onnx_optimization_flags: Vec<String>,

    /// Opt into the default-off alpha-beta-CROWN VGG treatment:
    /// exact 2x2 MaxPool decomposition plus property-size policy.
    ///
    /// This is intentionally model/preset scoped rather than a global solver
    /// default. Ineligible MaxPool nodes remain unchanged.
    pub(crate) vgg_abcrown_treatment: Option<bool>,
}

/// General configuration options.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct GeneralPreset {
    /// Root path for benchmark data (relative to config file).
    pub(crate) root_path: Option<PathBuf>,

    /// CSV file with instance list.
    pub(crate) csv_name: Option<String>,

    /// Compute device (cpu, wgpu).
    pub(crate) device: Option<String>,

    /// Loss reduction function (sum, max, min).
    pub(crate) loss_reduction_func: Option<String>,

    /// Convolution backward mode: auto, patches, or matrix.
    /// Reference: alpha-beta-CROWN `general.conv_mode` (`abcrown.py:228-231`).
    pub(crate) conv_mode: Option<ny_propagate::ConvMode>,

    /// Complete verifier selection: "auto", "bab", or "mip".
    /// Reference: alpha-beta-CROWN `general.complete_verifier`. Categories whose
    /// nets are MIP-exact and CROWN-loose (sat_relu, malbeware) route straight to
    /// the MIP solver with the full budget instead of burning it in BaB first.
    /// An explicit `--complete-verifier` CLI choice still wins over the preset.
    pub(crate) complete_verifier: Option<String>,
}

/// PGD attack configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct AttackPreset {
    /// PGD enablement. Reference alpha-beta-CROWN schedules PGD with
    /// "before"/"middle"/"after"; ny honors only enablement plus one schedule
    /// discriminator: "skip"/"none"/"disabled" disable PGD, "input_bab"
    /// enables PGD but suppresses the upfront stage, and
    /// "before"/"middle"/"after" all enable PGD on the upfront schedule
    /// (`apply_preset` warns for "middle"/"after", whose reference ordering
    /// is not implemented).
    pub(crate) pgd_order: Option<String>,
    /// Number of PGD restarts.
    pub(crate) pgd_restarts: Option<usize>,
    /// Number of PGD steps per restart.
    pub(crate) pgd_steps: Option<usize>,
    /// PGD alpha/step size. Accepts numeric YAML values or the string `auto`.
    #[serde(default, deserialize_with = "value_parse::option_string_or_number")]
    pub(crate) pgd_alpha: Option<String>,
    /// Whether `pgd_alpha` should scale by the input range.
    pub(crate) pgd_alpha_scale: Option<bool>,
    /// Per-step exponential decay for the PGD/Adam learning rate.
    /// Maps to `BetaCrownConfig::pgd_lr_decay` (→ `AdamClippingParams::lr_decay`).
    /// Reference: alpha-beta-CROWN `attack.pgd_lr_decay`.
    pub(crate) pgd_lr_decay: Option<f32>,
    pub(crate) attack_tolerance: Option<f32>,
    /// Restart PGD when a projected step leaves the point unchanged (#4278).
    pub(crate) pgd_restart_when_stuck: Option<bool>,
    /// Attack mode: "PGD" or "diversed_PGD" (OSI init, #1449).
    pub(crate) attack_mode: Option<String>,
    /// OSI initialization steps; only with `attack_mode: diversed_PGD` (#1449).
    pub(crate) osi_steps: Option<usize>,
    /// Straight-through-estimator surrogate gradient for Sign layers during
    /// ATTACK gradient estimation (#surrogate-sign). For binarized nets
    /// (traffic_signs QConv/Sign) the default tanh smooth relaxation
    /// saturates to a zero gradient; STE keeps the signal at any scale.
    pub(crate) surrogate_sign_gradient: Option<bool>,
    /// Dense deterministic grid sweep over low-effective-dimension input
    /// boxes as a pre-PGD attack phase (#dense-sweep).
    pub(crate) dense_low_dim_sweep: Option<bool>,
    /// Effective-dimension gate for the dense sweep (#dense-sweep).
    pub(crate) dense_sweep_max_dims: Option<usize>,
    /// Forward-evaluation budget for the dense sweep (#dense-sweep).
    pub(crate) dense_sweep_points: Option<usize>,
}

/// MIP complete-verifier configuration (`solver.mip` in alpha-beta-CROWN).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct MipPreset {
    /// MIP backend: "ay" (the only solver — SOLVER POLICY, ny-mip
    /// docs/SOLVER_POLICY.md). Legacy values "highs"/"scip" and
    /// alpha-beta-CROWN's "gurobi" resolve to ay with a warning.
    /// An explicit `--mip-solver` CLI choice wins.
    pub(crate) mip_solver: Option<String>,
    /// Number of parallel MIP solver processes/splits. Reserved for the
    /// phase-split racing mode (designs/scip.md Phase C); parsed for
    /// alpha-beta-CROWN key compatibility (`solver.mip.parallel_solvers`).
    pub(crate) parallel_solvers: Option<usize>,
}

/// Solver configuration (`solver:` in alpha-beta-CROWN).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct SolverPreset {
    /// Batch size for parallel domain processing.
    pub(crate) batch_size: Option<usize>,
    /// Maximum root spec rows per build batch (`solver.build_batch_size`).
    pub(crate) build_batch_size: Option<usize>,
    /// Automatically enlarge batch size based on GPU memory.
    pub(crate) auto_enlarge_batch_size: Option<bool>,

    /// Minimum batch size ratio when auto-enlarging.
    pub(crate) min_batch_size_ratio: Option<f32>,

    /// Bound propagation method: `crown`, `alpha-crown`, `forward+backward`, or `forward+crown`.
    /// Maps to `BetaCrownConfig::{use_alpha_crown,use_forward_bounds}` and rejects
    /// unsupported alpha-beta-CROWN modes instead of silently coercing to another setting.
    /// Alpha-beta-CROWN reference key: `solver.bound_prop_method`.
    #[serde(alias = "bound-prop-method")]
    pub(crate) bound_prop_method: Option<String>,

    /// α-CROWN configuration under solver (alpha-beta-CROWN naming).
    #[serde(default, alias = "alpha-crown")]
    pub(crate) alpha_crown: AlphaCrownPreset,

    /// β-CROWN configuration under solver (alpha-beta-CROWN naming).
    #[serde(default, alias = "beta-crown")]
    pub(crate) beta_crown: BetaCrownPreset,

    /// MIP complete-verifier configuration (alpha-beta-CROWN `solver.mip`).
    #[serde(default)]
    pub(crate) mip: MipPreset,
}

/// Branch-and-bound (BaB) configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct BabPreset {
    /// Batch size for parallel domain processing.
    pub(crate) batch_size: Option<usize>,

    /// Maximum number of output-adjacent backward layers/nodes for fixed-slope CROWN.
    pub(crate) crown_backward_layers: Option<usize>,

    /// Automatically enlarge batch size based on GPU memory.
    pub(crate) auto_enlarge_batch_size: Option<bool>,

    /// Minimum batch size ratio when auto-enlarging.
    pub(crate) min_batch_size_ratio: Option<f32>,

    /// Timeout in seconds.
    pub(crate) timeout: Option<u64>,

    /// Maximum domains to explore.
    pub(crate) max_domains: Option<usize>,

    /// Maximum tree depth.
    pub(crate) max_depth: Option<usize>,

    /// Early stopping patience (iterations without improvement).
    pub(crate) early_stop_patience: Option<usize>,

    /// Floor (seconds) for the graph CROWN-IBP collector's equal-share
    /// per-node time budget (#4413, #cgan-bn11-budget). Unset keeps the
    /// built-in 2.0 s constant.
    pub(crate) crown_ibp_per_node_floor_secs: Option<f64>,

    /// Cap (seconds) on the graph CROWN-IBP collector's equal-share per-node
    /// time budget (#4413, #cgan-bn11-budget). Unset keeps the built-in
    /// 12.0 s constant. cgan_2023 raises this to 150 s so the 28,800-dim
    /// BN_11 chunked backward (~143 s full collection, measured) can run
    /// in-pipeline instead of degrading to IBP.
    pub(crate) crown_ibp_per_node_cap_secs: Option<f64>,

    /// In-iteration verified-domain pruning (alpha-beta-CROWN
    /// `pruning_in_iteration`). Parsed for reference-config compatibility but
    /// NOT implemented — no engine code reads it; `apply_preset` warns and
    /// ignores it.
    pub(crate) pruning_in_iteration: Option<bool>,

    /// Enable intermediate bound transfer.
    pub(crate) interm_transfer: Option<bool>,

    /// Enable the one-time, structurally selected dense-head CROWN intermediate
    /// shrink-intersect at the graph root (#cifar-head-crown).
    pub(crate) root_crown_interm_dense_head: Option<bool>,

    /// Wall-clock cap in seconds for `root_crown_interm_dense_head`.
    pub(crate) root_crown_interm_max_secs: Option<u64>,

    /// Maximum selected dense-head pre-activation width.
    pub(crate) root_crown_interm_max_dim: Option<usize>,

    /// Enable the one-time structurally selected sparse crossing-row CROWN fold
    /// for non-dense ReLU pre-activations at the graph root.
    pub(crate) root_sparse_interm_crown: Option<bool>,

    /// Wall-clock cap in seconds for `root_sparse_interm_crown`.
    pub(crate) root_sparse_interm_crown_max_secs: Option<u64>,

    /// Maximum flattened target width for the sparse pass.
    pub(crate) root_sparse_interm_crown_max_dim: Option<usize>,

    /// Maximum crossing rows seeded per sparse target.
    pub(crate) root_sparse_interm_crown_max_rows: Option<usize>,

    /// Maximum sparse targets processed deepest-first.
    pub(crate) root_sparse_interm_crown_max_targets: Option<usize>,

    /// Enable the β/α-ascent graft on the multi-objective dense-spec lane
    /// (#mo-beta-graft): the wide GPU segment-lane ascent optimizes the split
    /// β/α multipliers and the tight dense-spec primitive evaluates with them
    /// folded in (elementwise-tightest composition). Env `NY_MO_BETA_GRAFT`
    /// overrides in both directions.
    pub(crate) beta_graft: Option<bool>,

    /// Branching configuration.
    #[serde(default)]
    pub(crate) branching: BranchingPreset,

    /// α-CROWN configuration overrides.
    /// Supports both "alpha_crown" and "alpha-crown" naming.
    #[serde(default, alias = "alpha-crown")]
    pub(crate) alpha_crown: AlphaCrownPreset,

    /// β-CROWN configuration overrides.
    /// Supports both "beta_crown" and "beta-crown" naming.
    #[serde(default, alias = "beta-crown")]
    pub(crate) beta_crown: BetaCrownPreset,

    /// Reject the easy-to-miss sibling spelling of the per-disjunct α knob.
    ///
    /// The implemented alpha-beta-CROWN-compatible location is
    /// `bab.beta_crown.optimize_disjuncts_separately`. Without this narrow trap,
    /// Serde silently ignores `bab.optimize_disjuncts_separately`, leaving the
    /// experiment off while the preset otherwise loads successfully. Keep the
    /// rest of `BabPreset` permissive for winner-config compatibility.
    #[serde(
        default,
        rename = "optimize_disjuncts_separately",
        alias = "optimize-disjuncts-separately",
        deserialize_with = "reject_misplaced_optimize_disjuncts_separately",
        skip_serializing
    )]
    #[allow(dead_code)]
    pub(crate) rejected_misplaced_optimize_disjuncts_separately: Option<bool>,

    /// GCP-CROWN cutting planes configuration.
    #[serde(default)]
    pub(crate) cuts: CutsPreset,

    /// INVPROP configuration (output constraint propagation).
    #[serde(default)]
    pub(crate) invprop: InvpropPreset,

    /// Clip-and-verify configuration.
    #[serde(default)]
    pub(crate) clip: ClipPreset,

    /// Phase-level time budget overrides (#2206).
    /// Only explicitly-set fields override `PhaseBudgetConfig` defaults.
    #[serde(default)]
    pub(crate) phase_budget: PhaseBudgetPreset,
}

/// Phase-level time budget preset overrides (#2206 Packet E).
///
/// Each field is `Option` so only explicitly-set values override the
/// `PhaseBudgetConfig` defaults. This follows the same pattern as other
/// preset structs (e.g., `ClipPreset`, `CutsPreset`).
///
/// Source fractions and their defaults are documented in
/// `ny_propagate::PhaseBudgetConfig`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct PhaseBudgetPreset {
    /// Fraction of the BaB budget for the iterative root alpha-CROWN warmup
    /// (the foundational IBP/CROWN-IBP node-bounds sweep is never capped).
    /// Default: 0.20. Recommended competitive: 0.15; set 1.0 to uncap.
    pub(crate) initial_bounds_fraction: Option<f32>,

    /// Fraction of total timeout for upfront PGD attack.
    /// Default: 0.20.
    pub(crate) upfront_pgd_fraction: Option<f32>,

    /// Fraction of total timeout for reduced verification (sequential path).
    /// Default: 0.40.
    pub(crate) reduced_verification_fraction: Option<f32>,

    /// Fraction of total timeout for disjunctive global PGD.
    /// Default: 0.50.
    pub(crate) disjunctive_pgd_fraction: Option<f32>,

    /// Fraction of total timeout for disjunctive CROWN/alpha precheck.
    /// Default: 0.20.
    pub(crate) disjunctive_precheck_fraction: Option<f32>,

    /// Minimum fraction of total timeout guaranteed for MIP fallback.
    /// Default: 0.25.
    pub(crate) mip_min_fraction: Option<f32>,

    /// Minimum MIP timeout in seconds (floor clamp).
    /// Default: 5.
    pub(crate) mip_min_secs: Option<u64>,

    /// Maximum MIP timeout in seconds (ceiling clamp).
    /// Default: 30.
    pub(crate) mip_max_secs: Option<u64>,

    /// Fraction of total timeout reserved for post-BaB PGD attack (BaB stops
    /// at `timeout * (1 - fraction)` so the fallback attack gets the rest).
    /// Default: 0.10. Set 0.0 to disable the reservation.
    pub(crate) post_bab_pgd_fraction: Option<f32>,

    /// Fraction of the REMAINING budget for the single adaptive attack
    /// extension (#attack-extend). Default: 0.15. Set 0.0 to disable for
    /// categories where the promising-margin gate cannot discriminate
    /// (e.g. cgan_2023 band properties).
    pub(crate) attack_extension_fraction: Option<f32>,
    /// Optional ABSOLUTE ceiling (seconds) on the disjunctive global PGD phase,
    /// on top of `disjunctive_pgd_fraction`. Default: None (pure fraction).
    /// Recommended for hold-heavy conv benchmarks (cifar100/tinyimagenet) where
    /// PGD beyond a few seconds is wasted and the seconds are better spent in
    /// BaB (which re-bases on remaining time).
    pub(crate) disjunctive_pgd_max_secs: Option<u64>,
}

/// Branching heuristic configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct BranchingPreset {
    /// Branching method: "width", "impact", "babsr", "fsb", "kfsb", "input", "relu".
    pub(crate) method: Option<String>,

    /// Number of candidates for FSB/kFSB.
    pub(crate) candidates: Option<usize>,

    /// Reduce operation for kFSB: "min", "max", "mean".
    pub(crate) reduceop: Option<String>,

    /// Arm the multi-objective wave-batched kFSB selector (#kfsb-multi). When
    /// `Some(true)`, sets `BetaCrownConfig::use_kfsb_multi_branching = true`.
    /// Measured Pareto on cifar100 (9/9 strictly better bounds, 0 regressions);
    /// scoped to the cifar100 presets only. Env `NY_MO_KFSB` overrides the
    /// resulting arming in either direction (kill switch `NY_MO_KFSB=0`).
    pub(crate) kfsb_multi: Option<bool>,

    /// Input-splitting SB tuning.
    #[serde(default)]
    pub(crate) input_split: InputSplitPreset,

    /// Nonlinear split configuration (for networks with nonlinear operations).
    #[serde(default)]
    pub(crate) nonlinear_split: NonlinearSplitPreset,
}

/// Input-splitting SB configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct InputSplitPreset {
    /// Enable input splitting for alpha-beta-CROWN-compatible preset imports.
    pub(crate) enable: Option<bool>,

    /// Coefficient clamp threshold for SB input split scoring.
    #[serde(alias = "coeff_thresh")]
    pub(crate) sb_coeff_thresh: Option<f32>,

    /// Bonus score for intervals that touch zero.
    pub(crate) touch_zero_score: Option<f32>,

    /// Margin weight for SB input split scoring.
    #[serde(alias = "margin_weight")]
    pub(crate) sb_margin_weight: Option<f32>,

    /// Sum across spec rows instead of taking the max.
    pub(crate) sb_sum: Option<bool>,

    /// Restrict SB scoring to a single specification row.
    #[serde(alias = "primary_spec")]
    pub(crate) sb_primary_spec: Option<usize>,

    /// Enable IBP enhancement for the input-split BaB loop.
    /// When true, each domain (root + children) is screened with fast IBP before
    /// expensive CROWN backward. Domains verified by IBP skip CROWN entirely.
    /// Reference: alpha-beta-CROWN `bab.branching.input_split.ibp_enhancement`
    pub(crate) ibp_enhancement: Option<bool>,

    /// Enable the domain-stacked dense-spec batched rebound
    /// (#cgan-batched-stack): one conv/BN backward call per node across the
    /// whole domain batch, plus fresh per-domain IBP re-anchoring when
    /// `ibp_enhancement` is also set. ny-specific (no alpha-beta-CROWN
    /// counterpart; the reference batches all layers on GPU natively).
    pub(crate) stacked_rebound: Option<bool>,

    /// Parallelize independent per-domain warm-alpha refinements in the
    /// deferred reordered rebound. ny-specific and default false; a preset must
    /// opt in before `NY_INPUT_SPLIT_WARM_PARALLEL` may select the parallel arm.
    pub(crate) warm_parallel: Option<bool>,

    /// Use reordered BaB loop: bound before split (bound → filter → split → clip).
    /// Reference: alpha-beta-CROWN `bab.branching.input_split.reorder_bab`
    pub(crate) reorder_bab: Option<bool>,

    /// Domain-count threshold for adversarial checking during BaB.
    /// -1 = disabled, 0 = from first iteration, N = after N domains explored.
    /// Reference: alpha-beta-CROWN `bab.branching.input_split.adv_check`
    pub(crate) adv_check: Option<i32>,

    /// Number of input dimensions to split per parent (multi-dimensional input
    /// split). Each parent is midpoint-split on the top-`depth` SB-scored dims,
    /// producing up to 2^depth children that exactly cover the parent (BaB
    /// completeness preserved). Default 1 = the classic 1-dim → 2-child split.
    /// Larger values fill the GPU batch from fewer parents.
    /// Reference: alpha-beta-CROWN `storage_depth` (fills `batch_size`).
    #[serde(alias = "storage_depth")]
    pub(crate) depth: Option<usize>,

    /// Per-sub-domain α refinement iterations in the input-split BaB loop.
    /// When > 0 AND alpha-CROWN is enabled, each sub-domain warm-starts from its
    /// parent's optimized alphas and re-optimizes them for this many SPSA
    /// iterations against the sub-domain's tighter box. Default 0 (off).
    /// Reference: alpha-beta-CROWN `solver.alpha-crown.input_split_alpha_iteration`.
    #[serde(alias = "input_split_alpha_iteration")]
    pub(crate) alpha_iteration: Option<usize>,

    /// Learning rate for per-sub-domain α refinement (see `alpha_iteration`).
    /// Only used when `alpha_iteration > 0`. Default 0.05.
    /// Reference: alpha-beta-CROWN `solver.alpha-crown.input_split_lr_alpha`.
    #[serde(alias = "input_split_lr_alpha")]
    pub(crate) lr_alpha: Option<f32>,
}

/// Nonlinear split configuration.
///
/// In alpha-beta-CROWN this section selects the GenBaB branching path for networks
/// with general nonlinearities (bounded `Mul`/`MatMul`, Sigmoid, Sin/Cos, …). ny
/// treats a configured `nonlinear_split` (any field present, or `enable: true`) as a
/// request for [`ny_propagate::BranchingHeuristic::GenBaB`], so the BaB loop splits
/// the product / activation inputs and tightens the McCormick relaxation, rather than
/// falling back to pure input splitting which cannot touch the nonlinear frontier.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct NonlinearSplitPreset {
    /// Explicitly enable GenBaB nonlinear branching. When unset, GenBaB is still
    /// selected if `filter` / `filter_beta` are present and no `method` is pinned.
    pub(crate) enable: Option<bool>,

    /// Enable filtering for nonlinear splits.
    pub(crate) filter: Option<bool>,

    /// Enable beta filtering for nonlinear splits.
    pub(crate) filter_beta: Option<bool>,
}

impl NonlinearSplitPreset {
    /// Whether this preset section requests GenBaB nonlinear branching.
    ///
    /// True when explicitly enabled, or when any nonlinear-split tuning field is
    /// present (mirrors alpha-beta-CROWN, where a populated `nonlinear_split`
    /// section is itself the GenBaB directive). A `Some(false)` `enable` opts out.
    pub(crate) fn requests_genbab(&self) -> bool {
        match self.enable {
            Some(enabled) => enabled,
            None => self.filter.is_some() || self.filter_beta.is_some(),
        }
    }
}

/// α-CROWN configuration overrides.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct AlphaCrownPreset {
    /// Learning rate for α parameters.
    pub(crate) lr_alpha: Option<f32>,

    /// Learning rate decay factor (exponential).
    pub(crate) lr_decay: Option<f32>,

    /// Number of optimization iterations.
    /// Supports both "iterations" (ny) and "iteration" (alpha-beta-CROWN).
    #[serde(alias = "iteration")]
    pub(crate) iterations: Option<usize>,

    /// Share α parameters across batch.
    pub(crate) share_alphas: Option<bool>,

    /// Softmax bound mode. `"complex"` decomposes each Softmax node into the
    /// alpha-optimizable Exp/ReduceSum/Reciprocal/MulBinary primitive subgraph
    /// at model load (vit_2023). Analog of alpha-beta-CROWN's
    /// `bound_opts={'softmax': 'complex'}` (vnncomp23 vit winner recipe,
    /// `custom_adhoc_tuning.py`). Any other value warns and keeps the default
    /// direct-LSE softmax relaxation. Runtime kill-switch:
    /// `NY_NO_SOFTMAX_COMPLEX=1`.
    pub(crate) softmax: Option<String>,

    /// Use full convolution alpha (memory intensive).
    pub(crate) full_conv_alpha: Option<bool>,
    /// Skip saving best bounds during warmup. Matches α,β-CROWN's `start_save_best`.
    pub(crate) start_save_best: Option<f32>,
}

/// β-CROWN configuration overrides.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct BetaCrownPreset {
    /// Learning rate for α parameters during β-CROWN.
    pub(crate) lr_alpha: Option<f32>,

    /// Learning rate for β parameters.
    pub(crate) lr_beta: Option<f32>,

    /// Learning rate decay factor.
    pub(crate) lr_decay: Option<f32>,

    /// Number of optimization iterations.
    /// Supports both "iterations" (ny) and "iteration" (alpha-beta-CROWN).
    #[serde(alias = "iteration")]
    pub(crate) iterations: Option<usize>,

    /// Maximum depth for β optimization.
    pub(crate) max_depth: Option<usize>,
    pub(crate) optimize_disjuncts_separately: Option<bool>, // #4355
}

/// GCP-CROWN cuts configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct CutsPreset {
    /// Enable cutting planes.
    pub(crate) enabled: Option<bool>,

    /// Maximum number of cuts to maintain.
    pub(crate) max_cuts: Option<usize>,

    /// Minimum depth for cut generation.
    pub(crate) min_cut_depth: Option<usize>,

    /// Enable near-miss cut generation.
    pub(crate) near_miss: Option<bool>,

    /// Near-miss margin threshold.
    pub(crate) near_miss_margin: Option<f32>,

    /// Enable proactive cut generation (BICCOS-lite).
    pub(crate) proactive: Option<bool>,

    /// Maximum proactive cuts.
    pub(crate) max_proactive: Option<usize>,
}

/// INVPROP configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct InvpropPreset {
    /// Node names to apply output constraints to.
    pub(crate) apply_output_constraints_to: Option<Vec<String>>,

    /// Share ny parameters.
    pub(crate) share_gammas: Option<bool>,
}

/// Clip-and-verify configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct ClipPreset {
    /// Enable relaxed clipping.
    pub(crate) relaxed: Option<bool>,

    /// Relaxed clipping iterations.
    pub(crate) relaxed_iterations: Option<usize>,

    /// Clip type: "relaxed" (default) or "complete" (LP-optimal via Lagrangian dual).
    /// Reference: alpha-beta-CROWN `clip_input_domain.clip_type`
    pub(crate) clip_type: Option<String>,

    /// Fraction of unstable neurons for complete clipping neuron selection.
    /// -1.0 = all neurons (default). Only used with clip_type: complete.
    /// Reference: alpha-beta-CROWN `clip_neuron_selection_type` + `clip_neuron_selection_value`
    pub(crate) neuron_selection_ratio: Option<f32>,
    /// Enable intermediate domain clipping.
    pub(crate) interm_domain: Option<bool>,

    /// Top-k neurons for intermediate clipping.
    pub(crate) interm_topk: Option<usize>,

    /// Apply clipping during α-CROWN.
    pub(crate) in_alpha_crown: Option<bool>,
    /// Enable pruning of infeasible domains.
    pub(crate) prune: Option<bool>,

    /// Use final layer constraints for pruning.
    pub(crate) use_final_layer: Option<bool>,
}

/// Load a preset configuration from a YAML file.
pub(crate) fn load_preset(path: &Path) -> Result<PresetConfig> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("Failed to read preset config: {}", path.display()))?;

    let preset: PresetConfig = serde_yaml::from_str(&contents)
        .with_context(|| format!("Failed to parse preset YAML: {}", path.display()))?;

    Ok(preset)
}

pub(crate) fn build_onnx_load_config(preset: &PresetConfig) -> Result<ny_onnx::OnnxLoadConfig> {
    let flags = resolve_onnx_optimization_flags(preset)?;
    Ok(ny_onnx::OnnxLoadConfig::default().with_optimization_flags(flags))
}

pub(crate) fn resolve_onnx_optimization_flags(
    preset: &PresetConfig,
) -> Result<Vec<ny_onnx::OnnxOptimizationFlag>> {
    preset
        .model
        .onnx_optimization_flags
        .iter()
        .map(|flag| match normalize_flag_name(flag).as_str() {
            "merge_linear" => Ok(ny_onnx::OnnxOptimizationFlag::MergeLinear),
            _ => anyhow::bail!(
                "unsupported model.onnx_optimization_flags entry '{flag}': ny currently supports only 'merge_linear'"
            ),
        })
        .collect()
}

fn normalize_flag_name(flag: &str) -> String {
    flag.trim().to_ascii_lowercase().replace('-', "_")
}

fn string_or_vec_string<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        Single(String),
        Multiple(Vec<String>),
    }

    match StringOrVec::deserialize(deserializer)? {
        StringOrVec::Single(value) => Ok(vec![value]),
        StringOrVec::Multiple(values) => Ok(values),
    }
}

fn reject_misplaced_optimize_disjuncts_separately<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<bool>, D::Error>
where
    D: Deserializer<'de>,
{
    // Consume the value first so every YAML type gets the same actionable
    // location error instead of an unrelated bool type error.
    serde::de::IgnoredAny::deserialize(deserializer)?;
    Err(<D::Error as serde::de::Error>::custom(
        "misplaced bab.optimize_disjuncts_separately; use \
         bab.beta_crown.optimize_disjuncts_separately",
    ))
}
