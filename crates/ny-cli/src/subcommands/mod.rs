// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Command-line subcommand definitions.
//!
//! The `Commands` enum (clap `#[derive(Subcommand)]`) must remain in one file
//! because clap requires the full enum for derive. Shared CLI types are
//! extracted to `cli_types` to reduce coupling.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::commands;

pub(crate) mod cli_types;
pub(crate) mod whisper_args;

#[cfg(test)]
mod tests;

// Re-export shared CLI types at this module level for backwards compatibility
pub(crate) use cli_types::{
    AlphaGradientMethodArg, AlphaOptimizerArg, BackendArg, CompleteVerifierArg, LayerNormModeArg,
    LayerNormNormModeArg, LogFormat, MipSolverArg, MulBinaryRelaxationArg,
};
pub(crate) use whisper_args::WhisperCommonArgs;

#[derive(Parser)]
#[command(name = "ny")]
#[command(author = "ny Team")]
#[command(version = "0.1.0")]
#[command(about = "Neural network verification and analysis CLI", long_about = None)]
pub(crate) struct Cli {
    /// Verbosity level (-v info, -vv debug, -vvv trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Log output format (text or json)
    #[arg(long, value_enum, default_value_t = LogFormat::Text)]
    log_format: LogFormat,

    #[command(subcommand)]
    command: Commands,
}

impl Cli {
    /// Split CLI into ownership-friendly parts for main dispatch.
    /// Fields remain private to avoid ad-hoc access outside this module.
    #[must_use]
    pub(crate) fn into_parts(self) -> (u8, LogFormat, Commands) {
        (self.verbose, self.log_format, self.command)
    }
}

#[derive(clap::Args)]
pub(crate) struct VerifyArgs {
    /// Path to model (ONNX, SafeTensors, PyTorch, GGUF, NNet)
    #[arg(value_name = "MODEL")]
    pub(crate) model: Option<PathBuf>,

    /// Path to model (backwards-compatible flag; prefer positional MODEL)
    #[arg(long = "model", value_name = "MODEL", conflicts_with = "model")]
    pub(crate) model_flag: Option<PathBuf>,

    /// YAML config file (alpha-beta-CROWN compatible)
    #[arg(long, value_name = "PATH")]
    pub(crate) config: Option<PathBuf>,

    /// Root path for config inference (uses <root>/config.yaml if --config is omitted)
    #[arg(long, value_name = "PATH")]
    pub(crate) root_path: Option<PathBuf>,

    /// Input perturbation epsilon (ignored if --property is specified;
    /// fallback: 0.01 when config does not provide a value)
    #[arg(short, long)]
    pub(crate) epsilon: Option<f32>,

    /// VNN-LIB property file (.vnnlib) specifying input bounds and output constraints
    #[arg(short, long)]
    pub(crate) property: Option<PathBuf>,

    /// Peel off terminal Softmax/LogSoftmax/Sigmoid when constraints are logit comparisons (true/false).
    /// Requires --property (VNN-LIB).
    #[arg(long, value_parser = clap::value_parser!(bool))]
    pub(crate) peel_off_last_softmax_layer: Option<bool>,

    /// Verification method (ibp, crown, alpha, sdp, beta; default: alpha)
    #[arg(long)]
    pub(crate) method: Option<String>,

    /// MulBinary relaxation mode for CROWN propagation
    #[arg(long, value_enum, default_value_t = MulBinaryRelaxationArg::Mccormick)]
    pub(crate) mul_binary_relaxation: MulBinaryRelaxationArg,

    /// Timeout in seconds (fallback: 60 when config does not provide a value)
    #[arg(short, long)]
    pub(crate) timeout: Option<u64>,

    /// Compute backend (cpu, wgpu; fallback: cpu when config does not
    /// set general.device or solver.propagation.use_gpu)
    #[arg(long, value_enum)]
    pub(crate) backend: Option<BackendArg>,

    /// Use GPU acceleration (deprecated, use --backend wgpu instead)
    #[arg(long, default_value_t = false, hide = true)]
    pub(crate) gpu: bool,

    /// Load as native format (PyTorch/SafeTensors/GGUF) instead of ONNX
    #[arg(long, default_value_t = false)]
    pub(crate) native: bool,

    /// Use conservative (sound) LayerNorm bounds (disables forward-mode stabilization for IBP)
    #[arg(long, default_value_t = false)]
    pub(crate) conservative_layernorm: bool,

    /// Normalization CROWN mode for LayerNorm/RMSNorm/GroupNorm/InstanceNorm1d/AdaIN1d
    #[arg(long, value_enum, default_value_t = LayerNormModeArg::Sound)]
    pub(crate) layernorm_mode: LayerNormModeArg,

    /// LayerNorm normalization mode: standard (default, full LayerNorm),
    /// or mean-only (DeepT-style, subtract mean without variance normalization)
    #[arg(long, value_enum, default_value_t = LayerNormNormModeArg::Standard)]
    pub(crate) layernorm_norm_mode: LayerNormNormModeArg,

    /// Layer-by-layer verification mode: outputs bound statistics per node
    /// Useful for large models where full verification may timeout
    #[arg(long, default_value_t = false, conflicts_with = "block_wise")]
    pub(crate) layer_by_layer: bool,

    /// Block-wise verification mode: resets bounds at each transformer block
    /// Prevents bound explosion and enables per-block zonotope tightening
    #[arg(long, default_value_t = false, conflicts_with = "layer_by_layer")]
    pub(crate) block_wise: bool,

    /// Show progress during verification (useful for large models)
    /// Works with `--block-wise` and `--layer-by-layer`.
    #[arg(long, default_value_t = false)]
    pub(crate) progress: bool,

    /// Output progress as JSON lines to stderr (for programmatic monitoring)
    /// Each line is a complete JSON object with progress information.
    /// Implies --progress. Works with `--block-wise` and `--layer-by-layer`.
    #[arg(long, default_value_t = false)]
    pub(crate) progress_json: bool,

    /// Maximum number of blocks to verify (0 = all blocks)
    /// Useful for partial verification of very large models
    /// Requires --block-wise or --checkpoint
    #[arg(long, default_value_t = 0)]
    pub(crate) max_blocks: usize,

    /// Checkpoint file for save/resume (implies --block-wise)
    /// If file exists and matches config, resume from checkpoint.
    /// Saves progress after each block for crash recovery.
    #[arg(long)]
    pub(crate) checkpoint: Option<PathBuf>,

    /// Output as JSON
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    /// Fail if verification method falls back (e.g., CROWN to IBP)
    /// By default, fallback occurs silently with actual_method reported.
    /// Use --strict to require the requested method succeeds or error.
    #[arg(long, default_value_t = false)]
    pub(crate) strict: bool,

    /// Fail if verification used any heuristics (not provably sound).
    /// Use this to ensure only sound verification results are accepted.
    /// If heuristics are used (e.g., sampling-based relaxations, forward-mode LayerNorm),
    /// the command exits non-zero with an error message.
    #[arg(long, default_value_t = false)]
    pub(crate) require_sound: bool,

    /// Allow heuristic sampling-based LogSoftmax CROWN relaxations (not provably sound).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_heuristic_logsoftmax: bool,

    /// Allow heuristic sampling-based Softmax/CausalSoftmax CROWN relaxations (not provably sound).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_heuristic_softmax: bool,

    /// Treat Unknown results as success (exit code 0).
    /// By default, Unknown returns exit code 2.
    /// Use this for CI on problems that may timeout or be undecidable.
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unknown: bool,

    /// Use f64 (double precision) for bound propagation.
    /// Required for soundnessbench/sat_relu. Only supports sequential Linear+Conv2D+ReLU.
    /// Reference: alpha-beta-CROWN `double_fp: true`.
    #[arg(long, default_value_t = false)]
    pub(crate) double_fp: bool,

    /// Shrink VNN-LIB input bounds inward by this epsilon (soundness defense).
    /// Each dimension's lower bound increases by eps and upper bound decreases by eps.
    /// Required for soundnessbench (`shrink_eps: 1e-10`).
    /// Reference: alpha-beta-CROWN `shrink_vnnlib` (`specifications.py:535-540`).
    #[arg(long, value_name = "EPS")]
    pub(crate) shrink_eps: Option<f64>,
}

#[derive(clap::Args)]
pub(crate) struct BenchArgs {
    /// Benchmark type (layer, attention, full, acasxu)
    #[arg(short, long, default_value = "layer")]
    pub(crate) benchmark: String,

    /// Output as JSON
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    /// VNN-COMP year (for acasxu benchmark: 2021, 2023, 2024, 2025)
    #[arg(long, default_value_t = 2021)]
    pub(crate) year: u32,

    /// Override timeout per problem in seconds (for acasxu benchmark)
    #[arg(long)]
    pub(crate) timeout: Option<u64>,

    /// Include individual problem results in JSON output (for acasxu benchmark)
    #[arg(
        long,
        default_value_t = true,
        value_parser = clap::value_parser!(bool),
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    pub(crate) include_results: bool,

    /// Filter to a specific ACAS-Xu model (e.g., "1_1" for ACASXU_run2a_1_1)
    #[arg(long)]
    pub(crate) model_filter: Option<String>,

    /// Filter to a specific ACAS-Xu property (e.g., "1" for prop_1.vnnlib)
    #[arg(long)]
    pub(crate) property_filter: Option<String>,

    /// Branching heuristic (for acasxu benchmark: width, impact, babsr, fsb, kfsb, kfsb-intercept-only, sequential, input).
    /// Default "input" matches α,β-CROWN config and achieves 98%+ pass rate.
    /// With `--gpu-bab`, only `impact`/`babsr` is currently supported.
    #[arg(long, default_value = "input")]
    pub(crate) branching: String,

    /// Maximum number of domains to explore (for acasxu benchmark)
    #[arg(long, default_value_t = 10000)]
    pub(crate) max_domains: usize,

    /// Enable proactive cut generation (for acasxu benchmark, BICCOS-lite).
    /// Generates cuts for unstable ReLUs BEFORE BaB starts.
    /// Helps on hard instances where initial bounds are loose.
    #[arg(long, default_value_t = false)]
    pub(crate) proactive_cuts: bool,

    /// Maximum number of proactive cuts to generate (for acasxu benchmark, default: 100)
    #[arg(long)]
    pub(crate) max_proactive_cuts: Option<usize>,

    /// Enable relaxed clipping for input splitting (for acasxu benchmark, Clip-and-Verify).
    /// Critical for ACAS-Xu: enables input domain tightening using CROWN constraints.
    #[arg(long, default_value_t = false)]
    pub(crate) relaxed_clip: bool,

    /// Enable PGD attack to find counterexamples (for acasxu benchmark).
    /// Can quickly identify violations, improving "falsified" detection.
    #[arg(long, default_value_t = false)]
    pub(crate) pgd_attack: bool,

    /// Number of PGD attack restarts (for acasxu benchmark).
    /// Default: 100, but auto-increased to 10000 when --branching=input.
    #[arg(long)]
    pub(crate) pgd_restarts: Option<usize>,

    /// Use GPU-accelerated BaB with DomainList storage (for acasxu benchmark).
    /// Routes verification through verify_graph_gpu_domain_list instead of
    /// sequential verify(). Currently supports only:
    /// `--branching=impact` (alias: `--branching=babsr`).
    #[arg(long, default_value_t = false)]
    pub(crate) gpu_bab: bool,

    /// Disable lA warm-start in GPU BaB backward pass (for A/B benchmarking).
    /// When set, each backward pass recomputes from scratch instead of reusing
    /// cached linear bound coefficients from the parent domain.
    #[arg(long, default_value_t = false)]
    pub(crate) no_la_warm_start: bool,

    /// Compute backend (cpu, wgpu). Honored only by the full-suite CROWN
    /// microbenchmark (for layer/attention it is IBP-only; acasxu GPU
    /// opt-in is --gpu-bab).
    #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
    pub(crate) backend: BackendArg,

    /// Use wgpu GPU acceleration (deprecated, use --backend wgpu).
    /// Honored only by the full-suite CROWN microbenchmark.
    #[arg(long, default_value_t = false, hide = true)]
    pub(crate) gpu: bool,
}

#[derive(clap::Args)]
pub(crate) struct BetaCrownArgs {
    /// Path to ONNX model
    pub(crate) model: PathBuf,

    /// VNN-LIB property file (.vnnlib) specifying input bounds and output constraints
    #[arg(short, long)]
    pub(crate) property: Option<PathBuf>,

    /// Preset configuration file (YAML) with benchmark-specific tuning.
    /// Preset values are overridden by explicit CLI flags.
    /// Example: --preset configs/vnncomp24/cifar100.yaml
    #[arg(long)]
    pub(crate) preset: Option<PathBuf>,

    /// Input perturbation epsilon (ignored if --property is specified)
    #[arg(short, long, default_value = "0.01")]
    pub(crate) epsilon: f32,

    /// Property threshold: verify that output > threshold (ignored if --property is specified)
    #[arg(short, long, default_value = "0.0")]
    pub(crate) threshold: f32,

    /// Peel off terminal Softmax/LogSoftmax/Sigmoid when constraints are logit comparisons.
    /// Requires --property (VNN-LIB).
    #[arg(long, default_value_t = false)]
    pub(crate) peel_off_last_softmax_layer: bool,

    /// Allow heuristic LogSoftmax relaxation (not provably sound).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_heuristic_logsoftmax: bool,

    /// Allow heuristic Softmax/CausalSoftmax relaxation (not provably sound).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_heuristic_softmax: bool,

    /// Maximum number of domains to explore (default: 10000, or preset value)
    #[arg(long)]
    pub(crate) max_domains: Option<usize>,

    /// Timeout in seconds (default: 300, or preset value)
    #[arg(long)]
    pub(crate) timeout: Option<u64>,

    /// Maximum search tree depth (number of splits) (default: 100, or preset value)
    #[arg(long)]
    pub(crate) max_depth: Option<usize>,

    /// Branching heuristic: auto (default), width, impact/babsr,
    /// fsb/kfsb/kfsb-intercept-only, sequential, input, relu.
    /// `auto` selects input-splitting vs ReLU/kFSB from the loaded model +
    /// spec (low-dimensional input => input splitting; otherwise kFSB; MIP
    /// complete-verifier categories use kFSB). An explicit value or a preset's
    /// `bab.branching.method` overrides auto.
    #[arg(long, default_value = "auto")]
    pub(crate) branching: Option<String>,

    /// Number of candidate neurons to evaluate for FSB branching (default: 8, or preset value)
    #[arg(long)]
    pub(crate) fsb_candidates: Option<usize>,

    /// Disable α-CROWN optimization (use CROWN-IBP only). Faster but looser bounds.
    #[arg(long, default_value_t = false)]
    pub(crate) no_alpha: bool,

    /// Number of α-CROWN optimization iterations (higher = tighter bounds, slower)
    /// For DAG models, each iteration = 1 + 2*spsa_samples CROWN passes.
    /// Default: 20 iterations (matches α,β-CROWN's init_iteration default), or preset value.
    #[arg(long)]
    pub(crate) alpha_iterations: Option<usize>,

    /// Per-sub-domain α refinement iterations in the input-split BaB loop
    /// (default: 0 = off, or preset value).
    ///
    /// When > 0 AND α-CROWN is enabled, each input-split sub-domain
    /// warm-starts from its parent's optimized alphas and re-optimizes them
    /// for this many SPSA iterations against the sub-domain's tighter box,
    /// producing tighter per-domain bounds (fewer splits). Has no effect
    /// under plain CROWN (`--no-alpha`).
    /// Reference: alpha-beta-CROWN `solver.alpha-crown.input_split_alpha_iteration`.
    #[arg(long)]
    pub(crate) input_split_alpha_iterations: Option<usize>,

    /// Learning rate for per-sub-domain α refinement (default: 0.05, or
    /// preset value). Only used when `--input-split-alpha-iterations > 0`.
    /// Reference: alpha-beta-CROWN `solver.alpha-crown.input_split_lr_alpha`.
    #[arg(long)]
    pub(crate) input_split_lr_alpha: Option<f32>,

    /// Disable adaptive α-CROWN skipping for deep networks.
    /// By default, α-CROWN is automatically skipped for very deep networks (>25 ReLU layers)
    /// where optimization doesn't help because bounds are fundamentally loose.
    /// Use this flag to always run α-CROWN regardless of network depth.
    #[arg(long, default_value_t = false)]
    pub(crate) no_adaptive_alpha_skip: bool,

    /// Depth threshold for adaptive α-CROWN skipping (number of ReLU layers).
    /// Networks with more than this many ReLU layers will skip α-CROWN if adaptive skip is enabled.
    /// Default: 8 (ResNet-2b has 6 and benefits, ResNet-4b has 10 and doesn't benefit), or preset value.
    #[arg(long)]
    pub(crate) alpha_skip_depth: Option<usize>,

    /// Use CROWN-IBP bounds for intermediate nodes (O(N²) but tighter).
    /// By default, uses IBP bounds (O(N), faster but looser - 3000x+ expansion).
    /// This matches α,β-CROWN's fix_interm_bounds=False setting.
    /// WARNING: Can be slow for deep networks (ResNet-4b: ~80s init vs <5s with IBP).
    #[arg(long, default_value_t = false)]
    pub(crate) crown_ibp_intermediates: bool,

    /// Number of SPSA samples per α-CROWN iteration (default: 1, or preset value).
    /// Higher values reduce gradient variance at the cost of more CROWN passes.
    /// Each sample requires 2 CROWN passes (plus/minus perturbation).
    /// Formula: total_passes = iterations * (1 + 2 * samples)
    #[arg(long)]
    pub(crate) alpha_spsa_samples: Option<usize>,

    /// Learning rate for α-CROWN optimization.
    /// Higher values converge faster but may overshoot. Lower values are more stable.
    /// Default 0.1 matches α,β-CROWN for Adam optimizer, or use preset value.
    #[arg(long)]
    pub(crate) alpha_lr: Option<f32>,

    /// Gradient method for α-CROWN optimization (default: analytic-chain).
    /// - analytic-chain: True chain-rule gradients (default; closest to
    ///   reference α,β-CROWN's loss.backward())
    /// - spsa: SPSA zero-order optimization (O(1) passes per iteration;
    ///   noise-dominated for networks with many unstable neurons)
    /// - fd: Finite differences (O(n) passes, accurate but slow)
    /// - analytic: Experimental - local gradients from CROWN backward (incomplete)
    #[arg(long, value_enum)]
    pub(crate) alpha_gradient_method: Option<AlphaGradientMethodArg>,

    /// Optimizer for α-CROWN parameter updates (default: adam, or preset value).
    /// - adam: Adam optimizer (adaptive moment estimation - matches α,β-CROWN)
    /// - sgd: SGD with momentum
    #[arg(long, value_enum)]
    pub(crate) alpha_optimizer: Option<AlphaOptimizerArg>,

    /// Enable INVPROP (output constraint backward propagation).
    ///
    /// Gamma optimization is not yet implemented: output constraints are
    /// recorded but bounds are unchanged today.
    ///
    /// Requires `--property` (VNN-LIB) to provide output constraints.
    #[arg(long, default_value_t = false)]
    pub(crate) invprop: bool,

    /// Layer patterns to apply INVPROP output constraints to (repeatable).
    ///
    /// Examples:
    /// - `--invprop-apply all` (default when omitted)
    /// - `--invprop-apply BoundReLU`
    /// - `--invprop-apply /input.7`
    #[arg(long, requires = "invprop")]
    pub(crate) invprop_apply: Vec<String>,

    /// Share gammas across neurons to reduce memory (INVPROP).
    #[arg(long, default_value_t = false, requires = "invprop")]
    pub(crate) invprop_share_gammas: bool,

    /// Number of β-CROWN optimization iterations per domain (default: 0, or preset value).
    /// Per-domain optimization is expensive; default 0 for throughput.
    /// Use --beta-iterations 5 for single-objective verification where bound quality matters.
    #[arg(long)]
    pub(crate) beta_iterations: Option<usize>,

    /// Maximum depth for per-domain β optimization (default: 3, or preset value).
    /// Only applies when beta_iterations > 0. Domains deeper than this
    /// skip optimization and rely on inherited β values from warmup.
    #[arg(long)]
    pub(crate) beta_max_depth: Option<usize>,

    /// Learning rate for β optimization (default: 0.05, or preset value).
    /// α,β-CROWN default is 0.05.
    #[arg(long)]
    pub(crate) lr_beta: Option<f32>,

    /// Use CROWN-IBP for tighter intermediate bounds (~66% tighter than IBP). Enabled by default.
    #[arg(
        long,
        default_value_t = true,
        value_parser = clap::value_parser!(bool),
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    pub(crate) crown_ibp: bool,

    /// Batch size for parallel domain processing (default: 64 GPU-optimized, or preset value).
    /// 1 = sequential processing.
    #[arg(long)]
    pub(crate) batch_size: Option<usize>,

    /// Disable parallel child domain creation (default: enabled)
    #[arg(long, default_value_t = false)]
    pub(crate) sequential_children: bool,

    /// Enable GCP-CROWN cutting planes (improves verification even without generating cuts)
    #[arg(
        long,
        default_value_t = true,
        value_parser = clap::value_parser!(bool),
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    pub(crate) enable_cuts: bool,

    /// Disable GCP-CROWN cutting planes (for comparison/debugging)
    #[arg(long, default_value_t = false)]
    pub(crate) no_cuts: bool,

    /// Maximum number of cutting planes to retain (default: 1000, or preset value)
    #[arg(long)]
    pub(crate) max_cuts: Option<usize>,

    /// Minimum depth for cut generation (default: 2, or preset value)
    /// Deeper domains produce more specific cuts.
    #[arg(long)]
    pub(crate) min_cut_depth: Option<usize>,

    /// Enable near-miss cut generation (experimental)
    /// Generates cuts from domains close to verification but not quite verified.
    #[arg(long, default_value_t = false)]
    pub(crate) enable_near_miss_cuts: bool,

    /// Margin for near-miss cut generation (default: 0.1, or preset value)
    /// Fraction of threshold or absolute if threshold=0.
    #[arg(long)]
    pub(crate) near_miss_margin: Option<f32>,

    /// Enable proactive cut generation (BICCOS-lite).
    /// Generates cuts for unstable ReLUs BEFORE BaB starts.
    /// Helps on hard instances where initial bounds are loose.
    #[arg(long, default_value_t = false)]
    pub(crate) proactive_cuts: bool,

    /// Maximum number of proactive cuts to generate (default: 100, or preset value)
    #[arg(long)]
    pub(crate) max_proactive_cuts: Option<usize>,

    /// Enable BICCOS constraint strengthening for verified-domain cuts.
    #[arg(long, default_value_t = false)]
    pub(crate) biccos_constraint_strengthening: bool,

    /// Drop ratio for BICCOS constraint strengthening (default: 0.5, or preset value)
    /// Quantile over influence scores.
    #[arg(long)]
    pub(crate) biccos_drop_ratio: Option<f32>,

    /// Enable relaxed clipping for input splitting (Clip-and-Verify).
    #[arg(long, default_value_t = false)]
    pub(crate) relaxed_clip: bool,

    /// Relaxed clipping iterations per split (default: 1, or preset value)
    #[arg(long)]
    pub(crate) relaxed_clip_iterations: Option<usize>,

    /// Enable intermediate domain clipping for ReLU splitting.
    #[arg(long, default_value_t = false)]
    pub(crate) clip_interm_domain: bool,

    /// Number of objective neurons per layer to tighten (default: 3, or preset value)
    /// Used with clip-interm-domain.
    #[arg(long)]
    pub(crate) clip_interm_topk: Option<usize>,

    /// Apply clip_interm_domain during alpha-CROWN optimization.
    #[arg(long, default_value_t = false)]
    pub(crate) clip_in_alpha_crown: bool,

    /// Prune infeasible domains during activation-space clipping.
    #[arg(long, default_value_t = false)]
    pub(crate) clip_interm_prune: bool,

    /// Use final-layer constraints when pruning clipped domains.
    #[arg(long, default_value_t = false)]
    pub(crate) clip_interm_use_final_layer: bool,

    /// Enable static intermediate bound transfer for batched domains.
    #[arg(long, default_value_t = false)]
    pub(crate) interm_transfer: bool,

    /// Enable PGD attack to find counterexamples (DEFAULT: on).
    ///
    /// PGD only FINDS counterexamples, which are independently re-validated
    /// before a `sat` verdict is emitted, so it can never change a
    /// verified/unsat verdict — disabling it never makes an unsound result
    /// sound or vice-versa. Its time budget is capped (a fraction of the
    /// timeout) so it cannot eat the whole BaB budget. Because the flag
    /// defaults to on, a preset's `attack.pgd_order: skip` takes precedence
    /// over it (the flag cannot signal an explicit enable). Pass
    /// `--no-pgd-attack` (or `--pgd-attack=false`) to force PGD off for
    /// reproducibility/determinism.
    #[arg(
        long = "pgd-attack",
        num_args = 0..=1,
        require_equals = true,
        default_value_t = true,
        default_missing_value = "true",
        action = clap::ArgAction::Set
    )]
    pub(crate) pgd_attack: bool,

    /// Disable PGD falsification (overrides --pgd-attack). For reproducibility
    /// or deterministic A/B comparisons. Bounding/disabling PGD is sound.
    #[arg(long = "no-pgd-attack", default_value_t = false, action = clap::ArgAction::SetTrue)]
    pub(crate) no_pgd_attack: bool,

    /// Number of PGD attack restarts (default: 100, or preset value)
    #[arg(long)]
    pub(crate) pgd_restarts: Option<usize>,

    /// Number of PGD gradient steps per restart (default: 50, or preset value)
    #[arg(long)]
    pub(crate) pgd_steps: Option<usize>,

    /// Compute backend (cpu, wgpu). Falls back to preset general.device, then CPU.
    #[arg(long, value_enum)]
    pub(crate) backend: Option<BackendArg>,

    /// Use wgpu GPU acceleration (deprecated, use --backend wgpu)
    #[arg(long, default_value_t = false, hide = true)]
    pub(crate) gpu: bool,

    /// Write grouped input-split batch summaries to JSONL during direct runs.
    #[arg(long)]
    pub(crate) input_split_metrics_jsonl: Option<PathBuf>,

    /// Write shared graph domain-batch summaries to JSONL during direct runs.
    #[arg(long)]
    pub(crate) domain_batch_metrics_jsonl: Option<PathBuf>,

    /// Output as JSON
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    /// Use GPU-accelerated BaB with DomainList storage.
    /// Enables verify_graph_gpu_domain_list for efficient batched GPU processing.
    /// Requires --branching relu (ReLU splitting for graph models).
    #[arg(long, default_value_t = false)]
    pub(crate) gpu_bab: bool,

    /// Disable lA warm-start in GPU BaB backward pass (for A/B benchmarking).
    /// When set, each backward pass recomputes from scratch instead of reusing
    /// cached linear bound coefficients from the parent domain.
    #[arg(long, default_value_t = false)]
    pub(crate) no_la_warm_start: bool,

    /// Complete verifier method.
    /// - auto (default): run BaB, then auto-escalate to the MIP (ay)
    ///   complete verifier when BaB is inconclusive AND the network is
    ///   MIP-encodable (sequential Linear+ReLU within a size cap). Escalation
    ///   is sound: MIP can only turn unknown/timeout into a decided verdict.
    ///   Requires a build with the MIP lane (the first-party ay solver).
    ///   Without it, auto behaves like bab and mip falls back to
    ///   branch-and-bound.
    /// - bab: Branch-and-bound with β-CROWN bounds only (never escalates).
    /// - mip: Direct SMT encoding with Big-M ReLUs (VNN-COMP compatibility)
    ///   For small networks, "mip" may be faster than BaB.
    #[arg(long, value_enum, default_value_t = CompleteVerifierArg::Auto)]
    pub(crate) complete_verifier: CompleteVerifierArg,

    /// MIP solver backend (when --complete-verifier mip).
    /// - ay: the ay solver, exact QF_LRA Big-M encoding (the only
    ///   backend — SOLVER POLICY: all solving in ny happens on ay)
    ///
    /// Requires a build with the MIP lane (the first-party ay solver).
    /// Without it, auto behaves like bab and mip falls back to
    /// branch-and-bound. Legacy preset values (highs/scip/gurobi)
    /// resolve to ay with a warning.
    #[arg(long, value_enum)]
    pub(crate) mip_solver: Option<MipSolverArg>,

    /// Competition mode: maximise verify-rate within the wall-clock budget
    /// by turning OFF proof-carrying certificate emission and internal
    /// self-checks (which are ON by default). SOUND: this never weakens the
    /// soundness of the sat/unsat verdict — it only skips emitting the
    /// extra machine-checkable certificate artifact. The VNN-COMP entry
    /// point (`ny vnncomp`) sets this automatically.
    #[arg(long, default_value_t = false)]
    pub(crate) competition_mode: bool,

    /// Force-disable certificate emission even outside competition mode.
    #[arg(long, default_value_t = false)]
    pub(crate) no_certificate: bool,

    /// Emit the proof-carrying certificate sidecar to this path (forces
    /// emission ON regardless of --competition-mode). When omitted and
    /// certificates are enabled, the default path is `<model>.cert.json`.
    #[arg(long)]
    pub(crate) emit_certificate: Option<PathBuf>,

    /// Speed override: ALLOW the UNSOUND fast f32 GPU CROWN backward to decide
    /// verdicts. By DEFAULT the soundness gate is ENGAGED so every verdict-
    /// deciding CROWN bound takes a PROVEN-SOUND path (the sound GPU-resident
    /// backward — still GPU-accelerated — or the CPU f64+γ_n·S fallback). The
    /// unsound fast path can be *tighter than the true range* and flip a
    /// genuinely-violated instance to `Verified`/`unsat`, so this trades
    /// correctness for speed — DO NOT use it when the verdict must be trusted.
    /// No effect on CPU-only runs. Ignored under `--competition-mode` (always sound).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unsound_gpu_crown: bool,
}

// Boxing the largest argument groups bounds stack use in Clap's generated debug schema builder.
#[derive(Subcommand)]
pub(crate) enum Commands {
    /// Verify a neural network property
    Verify(Box<VerifyArgs>),

    /// Load and inspect a model
    Inspect {
        /// Path to model (ONNX, PyTorch .pt/.pth, SafeTensors)
        model: PathBuf,

        /// Load as native format (PyTorch/SafeTensors) instead of ONNX
        #[arg(long, default_value_t = false)]
        native: bool,

        /// Include static FLOP and activation-memory estimates.
        /// Requires an ONNX model with fully-known tensor shapes.
        #[arg(long, default_value_t = false)]
        cost: bool,

        /// Apply a timing calibration profile to the static cost estimate.
        /// Requires `--cost` and an ONNX model.
        #[arg(long, value_name = "PATH", requires = "cost")]
        timing_profile: Option<PathBuf>,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Print the trace-bridge soundness-coverage manifest: an exhaustive
    /// catalogue classifying all 123 ingestable ops as exact / sound /
    /// sound_but_loose / unsupported (data-dependent, refused). Classification
    /// is build-enforced: the classifier is a wildcard-free match, so a newly
    /// added op fails to compile until it is deliberately classified.
    Coverage {
        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// SOUND certified global Lipschitz upper bound (exact rational, fails
    /// closed outside the Linear/Conv/ReLU fragment), shown next to the
    /// optimistic spectral-norm estimate
    Lipschitz {
        /// Path to ONNX model
        model: PathBuf,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Compare two single-input ONNX models (for porting verification)
    Compare {
        /// Path to reference single-input ONNX model (e.g., PyTorch export)
        reference: PathBuf,

        /// Path to target single-input ONNX model (e.g., converted target export)
        target: PathBuf,

        /// Maximum allowed difference in output bounds (looser than diff because bounds
        /// comparison has inherent approximation error from different bound propagation paths)
        #[arg(short, long, default_value = "0.001")]
        tolerance: f32,

        /// Input perturbation epsilon
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// Verification method (ibp, crown, alpha; aliases: alpha-crown, alpha_crown)
        #[arg(short = 'm', long, default_value = "crown")]
        method: String,

        /// Compute backend (cpu, wgpu).
        #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
        backend: BackendArg,

        /// Use wgpu GPU acceleration (deprecated, use --backend wgpu)
        #[arg(long, default_value_t = false, hide = true)]
        gpu: bool,

        /// Print per-element comparison details
        #[arg(long, default_value_t = false)]
        verbose: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Find where two ONNX models diverge (layer-by-layer diff for porting)
    Diff {
        /// Path to first ONNX model (e.g., PyTorch export)
        model_a: PathBuf,

        /// Path to second ONNX model (e.g., CoreML/Metal export)
        model_b: PathBuf,

        /// Path to input data file (.npy format)
        #[arg(long)]
        input: Option<PathBuf>,

        /// Maximum allowed difference before flagging divergence (tighter than compare
        /// because diff compares exact inference values, not approximated bounds)
        #[arg(short, long, default_value = "1e-5")]
        tolerance: f32,

        /// Optional JSON/YAML mapping of layer names (model_a -> model_b)
        #[arg(long)]
        layer_map: Option<PathBuf>,

        /// Continue comparing after first divergence
        #[arg(
            long,
            default_value_t = true,
            value_parser = clap::value_parser!(bool),
            num_args = 0..=1,
            default_missing_value = "true"
        )]
        continue_after_divergence: bool,

        /// Enable root cause diagnosis (analyzes divergence patterns)
        #[arg(long, default_value_t = false)]
        diagnose: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Analyze layer sensitivity (which layers amplify input noise)
    Sensitivity {
        /// Path to ONNX model
        model: PathBuf,

        /// Input perturbation epsilon
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// Continue analysis after overflow
        #[arg(
            long,
            default_value_t = true,
            value_parser = clap::value_parser!(bool),
            num_args = 0..=1,
            default_missing_value = "true"
        )]
        continue_after_overflow: bool,

        /// Show only high-sensitivity layers (sensitivity > threshold)
        #[arg(long)]
        threshold: Option<f32>,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Check if ONNX layers can be safely quantized to float16/int8
    QuantizeCheck {
        /// Path to ONNX model
        model: PathBuf,

        /// Input perturbation epsilon
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// Continue analysis after overflow
        #[arg(
            long,
            default_value_t = true,
            value_parser = clap::value_parser!(bool),
            num_args = 0..=1,
            default_missing_value = "true"
        )]
        continue_after_overflow: bool,

        /// Check only float16 (skip int8 analysis)
        #[arg(long, default_value_t = false, conflicts_with = "int8_only")]
        float16_only: bool,

        /// Check only int8 (skip float16 analysis)
        #[arg(long, default_value_t = false, conflicts_with = "float16_only")]
        int8_only: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Profile bound width growth through the network
    ProfileBounds {
        /// Path to model (ONNX, SafeTensors, PyTorch, GGUF)
        model: PathBuf,

        /// Input perturbation epsilon
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// Continue analysis after overflow
        #[arg(
            long,
            default_value_t = true,
            value_parser = clap::value_parser!(bool),
            num_args = 0..=1,
            default_missing_value = "true"
        )]
        continue_after_overflow: bool,

        /// Show only layers with growth ratio above threshold
        #[arg(long)]
        threshold: Option<f32>,

        /// Load as native format (PyTorch/SafeTensors/GGUF) instead of ONNX
        #[arg(long, default_value_t = false)]
        native: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,

        /// Use zeros-centered input (for validation against Auto-LiRPA).
        /// Default is unit-variance input (±1 alternating) for realistic LayerNorm bounds.
        #[arg(long, default_value_t = false)]
        center_zeros: bool,
    },

    /// Verify a Whisper model component
    Whisper {
        /// Path to Whisper ONNX model
        model: PathBuf,

        /// Component to verify (encoder, attention)
        #[arg(short, long, default_value = "encoder")]
        component: String,

        /// Specific layer index (optional)
        #[arg(short, long)]
        layer: Option<usize>,

        /// Input perturbation epsilon
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Verify multiple Whisper encoder blocks sequentially (diagnostic tool)
    #[command(hide = true)]
    WhisperSeq {
        #[command(flatten)]
        common: WhisperCommonArgs,

        /// Input perturbation epsilon
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// Multi-block config preset: default, strict, diagnostic, sound-tight
        #[arg(long, default_value = "default")]
        mode: String,

        /// Override: terminate on NaN/Infinity (true/false)
        #[arg(long, value_parser = clap::value_parser!(bool))]
        terminate_on_overflow: Option<bool>,

        /// Override: continue after overflow by clamping (true/false)
        #[arg(long, value_parser = clap::value_parser!(bool))]
        continue_after_overflow: Option<bool>,

        /// Override: clamp value used when continuing after overflow
        #[arg(long)]
        overflow_clamp_value: Option<f32>,
    },

    /// Sweep epsilon for sequential verification (find stable ranges)
    #[command(hide = true)]
    WhisperSweep {
        #[command(flatten)]
        common: WhisperCommonArgs,

        /// Minimum epsilon (inclusive)
        #[arg(long, default_value = "0.000001")]
        epsilon_min: f32,

        /// Maximum epsilon (inclusive)
        #[arg(long, default_value = "0.01")]
        epsilon_max: f32,

        /// Number of sweep points
        #[arg(long, default_value_t = 10)]
        steps: usize,

        /// Sweep linearly instead of logarithmically
        #[arg(long, default_value_t = false)]
        linear: bool,

        /// Multi-block config preset: default, strict, diagnostic, sound-tight (default: strict)
        #[arg(long, default_value = "strict")]
        mode: String,

        /// Print per-block width details for each epsilon point
        #[arg(long, default_value_t = false)]
        per_block: bool,
    },

    /// Binary-search for maximum epsilon that completes N blocks
    #[command(hide = true)]
    WhisperEpsSearch {
        #[command(flatten)]
        common: WhisperCommonArgs,

        /// Target number of blocks to complete without overflow/early termination
        #[arg(long)]
        target_blocks: Option<usize>,

        /// Minimum epsilon (inclusive, search lower bound)
        #[arg(long, default_value = "0.000001")]
        epsilon_min: f32,

        /// Maximum epsilon (inclusive, search upper bound)
        #[arg(long, default_value = "0.01")]
        epsilon_max: f32,

        /// Number of binary search iterations (default: 20 for ~1e-6 precision ratio)
        #[arg(long, default_value_t = 20)]
        iterations: usize,

        /// Multi-block config preset: default, strict, diagnostic, sound-tight (default: strict)
        #[arg(long, default_value = "strict")]
        mode: String,

        /// Show progress for each iteration
        #[arg(long, default_value_t = false)]
        verbose_search: bool,
    },

    /// Generate export script for PyTorch models
    Export {
        /// Model type (whisper, custom)
        #[arg(short, long, default_value = "whisper")]
        model_type: String,

        /// Model size (tiny, base, small, medium, large)
        #[arg(short, long, default_value = "tiny")]
        size: String,

        /// Output script path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Run benchmarks
    Bench(Box<BenchArgs>),

    /// Audit VNN-COMP benchmark category coverage
    #[command(hide = true)]
    VnncompAudit {
        /// VNN-COMP year (2021, 2023, 2024, 2025, 2026)
        #[arg(long, default_value_t = 2021)]
        year: u32,

        /// Test timeout per category in seconds
        #[arg(long, default_value_t = 30)]
        timeout: u64,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,

        /// Show verbose operator information
        #[arg(short, long, default_value_t = false)]
        verbose: bool,

        /// Filter to specific category
        #[arg(long)]
        category: Option<String>,
    },

    /// Download and inspect benchmark assets.
    Benchmarks {
        #[command(subcommand)]
        action: commands::vnncomp_benchmarks::BenchmarkAssetsAction,
    },

    /// Download VNN-COMP benchmark repositories.
    #[command(hide = true)]
    VnncompBenchmarks {
        /// VNN-COMP year to download. Repeat to download multiple years.
        #[arg(long = "year")]
        years: Vec<u32>,

        /// Include optional/historical years, including 2022.
        #[arg(long, default_value_t = false)]
        all: bool,

        /// Output final summary as JSON.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Build a VNN-COMP tool-submission tarball from this checkout.
    #[command(hide = true)]
    VnncompSubmit {
        /// Output tarball path.
        #[arg(short, long, default_value = "dist/ny-vnncomp-submission.tar.gz")]
        output: PathBuf,

        /// Skip the release binary build before packaging.
        #[arg(long, default_value_t = false)]
        no_build: bool,

        /// Validate and print the package plan without writing the tarball.
        #[arg(long, default_value_t = false)]
        dry_run: bool,

        /// Output as JSON.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Late-submit this tool to the VNN-COMP 2026 evaluation platform (all tracks).
    #[command(hide = true)]
    VnncompLateSubmit {
        #[command(subcommand)]
        action: commands::vnncomp_late_submit::LateSubmitAction,
    },

    /// Run VNN-COMP script-protocol tools over local benchmark instances.
    #[command(hide = true)]
    VnncompMatrix {
        /// VNN-COMP year to run.
        #[arg(long, default_value_t = 2026)]
        year: u32,

        /// Tool spec NAME=TOOL_DIR. Repeat to compare tools. Defaults to ny=.
        #[arg(long = "tool")]
        tools: Vec<String>,

        /// Category to include. Repeat to include multiple categories.
        #[arg(long = "category")]
        categories: Vec<String>,

        /// Evenly sample N instances per category. 0 runs all selected instances.
        #[arg(long, default_value_t = 0)]
        sample_per_category: usize,

        /// Limit total selected instances after sampling. 0 means no limit.
        #[arg(long, default_value_t = 0)]
        limit: usize,

        /// Override every instance timeout in seconds.
        #[arg(long)]
        timeout_override: Option<u64>,

        /// Skip prepare_instance.sh and only run run_instance.sh.
        #[arg(long, default_value_t = false)]
        skip_prepare: bool,

        /// Directory for result files, CSV, and JSON.
        #[arg(long, default_value = "reports/benchmarks/matrix")]
        output_dir: PathBuf,

        /// Print only final JSON summary.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Run β-CROWN branch-and-bound complete verification
    BetaCrown(Box<BetaCrownArgs>),

    /// Run a single VNN-COMP benchmark instance end-to-end (competition entry point).
    ///
    /// This is the native implementation of the `run_instance.sh` protocol: it
    /// auto-loads the category preset, computes the internal timeout tier, runs the
    /// β-CROWN verification with the AUTO defaults (branching/backend/complete-verifier/
    /// PGD are all self-selected — no strategy flags), translates the verdict to the
    /// VNN-COMP result string (unsat/sat/timeout/unknown/error), and writes RESULTS_FILE
    /// (first line = result; for `sat`, the SMT-LIB counterexample witness is appended).
    ///
    /// Protocol: `ny vnncomp v1 CATEGORY ONNX VNNLIB RESULTS_FILE TIMEOUT`.
    Vnncomp {
        /// Protocol version string; must be exactly "v1".
        version: String,

        /// Benchmark category (e.g. `acasxu_2023`, `cifar100_2024`). Drives preset auto-loading.
        category: String,

        /// Path to the ONNX model for this instance.
        onnx: PathBuf,

        /// Path to the VNN-LIB property file for this instance.
        vnnlib: PathBuf,

        /// File to write the one-line VNN-COMP result (+ witness body for `sat`).
        results_file: PathBuf,

        /// Scored competition budget, in seconds. Accepts fractional values
        /// (VNN-COMP instance CSVs carry budgets like `210.0`; metaroom_2023's
        /// whole column is fractional) — floored to whole seconds internally.
        #[arg(value_parser = parse_budget_secs)]
        timeout_secs: u64,

        /// Directory containing the `vnncomp*/{category}.yaml` presets.
        /// Defaults to auto-derivation from the binary/ONNX path (nearest ancestor
        /// `configs/` directory).
        #[arg(long)]
        configs_dir: Option<PathBuf>,
    },

    /// Inspect and compare model weights (SafeTensors, ONNX)
    Weights {
        #[command(subcommand)]
        action: commands::weights::WeightsAction,
    },

    /// Geometric ground-truth utilities: evaluate .gt.json specs and verify
    /// networks against analytic geometry (docs/GEOMETRIC_GROUND_TRUTH_PLAN.md)
    Gt {
        #[command(subcommand)]
        action: commands::gt::GtAction,
    },

    /// Learn to verify neural networks, interactively (start with no argument)
    Tutorial {
        #[command(subcommand)]
        topic: Option<commands::tutorial::TutorialTopic>,
    },
}

/// Parse a VNN-COMP scored budget that may be fractional (`210.0`) into whole
/// seconds (floored, minimum 1). Instance CSVs in the official benchmark sets
/// carry fractional budgets — a strict integer parser scored those instances
/// `error` before any verification ran.
fn parse_budget_secs(raw: &str) -> Result<u64, String> {
    let secs = raw
        .parse::<f64>()
        .map_err(|e| format!("invalid budget '{raw}': {e}"))?;
    if !secs.is_finite() || secs <= 0.0 {
        return Err(format!(
            "budget must be a positive number of seconds, got '{raw}'"
        ));
    }
    Ok((secs.floor() as u64).max(1))
}
