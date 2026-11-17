// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Command-line subcommand definitions.
//!
//! Shared argument types live in `cli_types` and `whisper_args`.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::commands;

pub(crate) mod cli_types;
pub(crate) mod whisper_args;

#[cfg(test)]
mod tests;

pub(crate) use cli_types::{
    AlphaGradientMethodArg, AlphaOptimizerArg, BackendArg, CompleteVerifierArg, LayerNormModeArg,
    LayerNormNormModeArg, LogFormat, MipSolverArg, MulBinaryRelaxationArg,
};
pub(crate) use whisper_args::WhisperCommonArgs;

#[derive(Parser)]
#[command(name = "ny")]
#[command(author = "ny Team")]
// #binary-identity: use the derived version; --build-id provides full provenance.
#[command(version = crate::build_id::DISPLAY_VERSION)]
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
    /// Path to model (ONNX, NNet, PyTorch, SafeTensors, CoreML, or GGUF)
    #[arg(value_name = "MODEL")]
    pub(crate) model: Option<PathBuf>,

    /// Path to model (backwards-compatible flag; prefer positional MODEL)
    #[arg(long = "model", value_name = "MODEL", conflicts_with = "model")]
    pub(crate) model_flag: Option<PathBuf>,

    /// YAML config file (alpha-beta-CROWN compatible)
    #[arg(long, value_name = "PATH")]
    pub(crate) config: Option<PathBuf>,

    /// Root path for config inference (uses `<root>/config.yaml` if --config is omitted)
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
    /// Ignored without --property (VNN-LIB).
    #[arg(long, value_parser = clap::value_parser!(bool))]
    pub(crate) peel_off_last_softmax_layer: Option<bool>,

    /// Verification method (ibp, crown, alpha, or beta; default: alpha).
    /// Beta requires an NNet (.nnet) model; use `beta-crown` for ONNX graphs.
    #[arg(long)]
    pub(crate) method: Option<String>,

    /// MulBinary relaxation mode for CROWN propagation
    #[arg(long, value_enum, default_value_t = MulBinaryRelaxationArg::Mccormick)]
    pub(crate) mul_binary_relaxation: MulBinaryRelaxationArg,

    /// Timeout in seconds (fallback: 60 when config does not provide a value)
    #[arg(short, long)]
    pub(crate) timeout: Option<u64>,

    /// Compute backend: cpu or wgpu
    ///
    /// wgpu is used only if the adapter passes its qualification checks;
    /// otherwise ny uses the CPU and reports why.
    #[arg(long, value_enum)]
    pub(crate) backend: Option<BackendArg>,

    /// Use GPU acceleration (deprecated, use --backend wgpu instead)
    #[arg(long, default_value_t = false, hide = true, conflicts_with = "backend")]
    pub(crate) gpu: bool,

    /// Load with the native importer (PyTorch, SafeTensors, CoreML, or GGUF)
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

    /// Report bound statistics for each node, for models where full
    /// verification would time out
    #[arg(long, default_value_t = false, conflicts_with = "block_wise")]
    pub(crate) layer_by_layer: bool,

    /// Reset bounds at each transformer block, to keep them from blowing up
    #[arg(long, default_value_t = false, conflicts_with = "layer_by_layer")]
    pub(crate) block_wise: bool,

    /// Show progress (with --block-wise or --layer-by-layer)
    #[arg(long, default_value_t = false)]
    pub(crate) progress: bool,

    /// Write progress to stderr as JSON lines (implies --progress)
    #[arg(long, default_value_t = false)]
    pub(crate) progress_json: bool,

    /// Verify at most this many blocks (0 = all; needs --block-wise or
    /// --checkpoint)
    #[arg(long, default_value_t = 0)]
    pub(crate) max_blocks: usize,

    /// Save progress to this file after each block, and resume from it when it
    /// matches the current run (implies --block-wise)
    #[arg(long)]
    pub(crate) checkpoint: Option<PathBuf>,

    /// Output as JSON
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    /// Fail instead of falling back to a weaker method (for example CROWN to
    /// IBP)
    ///
    /// Without --strict, a fallback is allowed and the method actually used is
    /// reported.
    #[arg(long, default_value_t = false)]
    pub(crate) strict: bool,

    /// Fail if the result relied on any heuristic (not provably sound) step,
    /// such as a sampling-based relaxation or forward-mode LayerNorm
    #[arg(long, default_value_t = false)]
    pub(crate) require_sound: bool,

    /// Allow heuristic sampling-based LogSoftmax CROWN relaxations (not provably sound).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_heuristic_logsoftmax: bool,

    /// Allow heuristic sampling-based Softmax/CausalSoftmax CROWN relaxations (not provably sound).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_heuristic_softmax: bool,

    /// Exit 0 on ordinary unknown results; refused witnesses stay 2, timeouts 3
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unknown: bool,

    /// Use f64 bound propagation (sequential Linear, Conv2d, and ReLU models
    /// only)
    #[arg(long, default_value_t = false)]
    pub(crate) double_fp: bool,

    /// Shrink every VNN-LIB input interval inward by EPS on both sides
    ///
    /// This verifies a smaller region than the property states.
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

    /// Branching heuristic for the acasxu benchmark (default: input)
    ///
    /// Values: width, impact, babsr, fsb, kfsb, kfsb-intercept-only, sequential,
    /// input. With --gpu-bab, use impact, babsr, or input.
    #[arg(long, default_value = "input")]
    pub(crate) branching: String,

    /// Maximum number of domains to explore (for acasxu benchmark)
    #[arg(long, default_value_t = 10000)]
    pub(crate) max_domains: usize,

    /// Generate cuts for unstable ReLUs before branch-and-bound starts (acasxu
    /// benchmark)
    #[arg(long, default_value_t = false)]
    pub(crate) proactive_cuts: bool,

    /// Maximum number of proactive cuts to generate (for acasxu benchmark, default: 100)
    #[arg(long)]
    pub(crate) max_proactive_cuts: Option<usize>,

    /// Tighten input-split domains with CROWN constraints (Clip-and-Verify;
    /// acasxu benchmark)
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
    /// Routes verification through verify_graph_gpu_domain_list.
    /// Supports `--branching=impact` (alias: `babsr`) and `--branching=input`.
    #[arg(long, default_value_t = false)]
    pub(crate) gpu_bab: bool,

    /// Recompute every GPU BaB backward pass from scratch instead of reusing the
    /// parent domain's linear bounds (for A/B benchmarking)
    #[arg(long, default_value_t = false)]
    pub(crate) no_la_warm_start: bool,

    /// Compute backend (cpu, wgpu). Honored only by the full-suite CROWN
    /// microbenchmark (for layer/attention it is IBP-only; acasxu GPU
    /// opt-in is --gpu-bab).
    #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
    pub(crate) backend: BackendArg,

    /// Use wgpu GPU acceleration (deprecated, use --backend wgpu).
    /// Honored only by the full-suite CROWN microbenchmark.
    #[arg(long, default_value_t = false, hide = true, conflicts_with = "backend")]
    pub(crate) gpu: bool,
}

#[derive(clap::Args)]
pub(crate) struct BetaCrownArgs {
    /// Path to model (ONNX or NNet)
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
    /// Ignored without --property (VNN-LIB).
    #[arg(long, default_value_t = false)]
    pub(crate) peel_off_last_softmax_layer: bool,

    /// Allow heuristic LogSoftmax relaxation (not provably sound).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_heuristic_logsoftmax: bool,

    /// Allow heuristic Softmax/CausalSoftmax relaxation (not provably sound).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_heuristic_softmax: bool,

    /// Maximum number of domains to explore (default: 100000, or preset value)
    #[arg(long)]
    pub(crate) max_domains: Option<usize>,

    /// Estimated graph BaB queue-payload budget in bytes.
    ///
    /// Zero means unlimited. The highest-priority domain is retained even if it
    /// alone exceeds the budget. If the cap evicts any unverified domain, the
    /// verifier returns unknown rather than claiming a proof from an
    /// incomplete search.
    #[arg(long)]
    pub(crate) max_queue_bytes: Option<usize>,

    /// Timeout in seconds (default: 300, or preset value).
    ///
    /// Zero is unbounded for BaB. MIP-only verification requires a positive
    /// timeout so solver and certificate phases have an authority deadline.
    #[arg(long)]
    pub(crate) timeout: Option<u64>,

    /// Maximum search tree depth (number of splits) (default: 100, or preset value)
    #[arg(long)]
    pub(crate) max_depth: Option<usize>,

    /// Branching heuristic (default: auto)
    ///
    /// Values: auto, width, impact (or babsr), fsb, kfsb, kfsb-intercept-only,
    /// sequential, input, relu. `auto` selects kFSB in MIP-complete mode;
    /// otherwise input splitting for low-dimensional inputs or small, shallow
    /// models with moderate inputs, and kFSB for remaining models. Explicit
    /// values and a preset's `bab.branching.method` override `auto`.
    #[arg(long, default_value = "auto")]
    pub(crate) branching: Option<String>,

    /// Number of candidate neurons to evaluate for FSB branching (default: 8, or preset value)
    #[arg(long)]
    pub(crate) fsb_candidates: Option<usize>,

    /// Disable α-CROWN optimization (plain CROWN: faster, looser bounds)
    #[arg(long, default_value_t = false)]
    pub(crate) no_alpha: bool,

    /// α-CROWN optimization iterations (default: 100, or preset value)
    ///
    /// More iterations give tighter bounds and take longer.
    #[arg(long)]
    pub(crate) alpha_iterations: Option<usize>,

    /// α refinement iterations per input-split sub-domain (default: 0 = off, or
    /// preset value)
    ///
    /// When positive and α-CROWN is on, each sub-domain starts from its
    /// parent's optimized α and re-optimizes it for this many SPSA iterations
    /// on its smaller box, which tightens bounds and reduces splitting. No
    /// effect with `--no-alpha`.
    #[arg(long)]
    pub(crate) input_split_alpha_iterations: Option<usize>,

    /// Learning rate for per-sub-domain α refinement (default: 0.05, or preset
    /// value). Used only when --input-split-alpha-iterations is positive.
    #[arg(long)]
    pub(crate) input_split_lr_alpha: Option<f32>,

    /// Always run α-CROWN, even on deep networks
    ///
    /// By default α-CROWN is skipped on networks with more than
    /// --alpha-skip-depth ReLU layers, where optimization rarely helps.
    #[arg(long, default_value_t = false)]
    pub(crate) no_adaptive_alpha_skip: bool,

    /// ReLU-layer count above which α-CROWN is skipped (default: 20, or preset
    /// value)
    #[arg(long)]
    pub(crate) alpha_skip_depth: Option<usize>,

    /// Use CROWN-IBP instead of IBP for intermediate bounds
    ///
    /// Can require substantial time and memory on large graphs. Run under an
    /// external memory limit; --max-queue-bytes does not cap this work.
    #[arg(long, default_value_t = false)]
    pub(crate) crown_ibp_intermediates: bool,

    /// SPSA samples per α-CROWN iteration (default: 1, or preset value)
    ///
    /// Used only with --alpha-gradient-method spsa. More samples reduce
    /// gradient noise; each costs two extra CROWN passes.
    #[arg(long)]
    pub(crate) alpha_spsa_samples: Option<usize>,

    /// Learning rate for α-CROWN optimization (default: 0.1, or preset value)
    #[arg(long)]
    pub(crate) alpha_lr: Option<f32>,

    /// Gradient method for α-CROWN optimization (default: analytic-chain)
    #[arg(long, value_enum)]
    pub(crate) alpha_gradient_method: Option<AlphaGradientMethodArg>,

    /// Optimizer for α-CROWN (default: adam, or preset value)
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

    /// β-CROWN optimization iterations per domain (default: 15, or preset value)
    ///
    /// Per-domain optimization is expensive; 0 turns it off for throughput.
    #[arg(long)]
    pub(crate) beta_iterations: Option<usize>,

    /// Deepest domain that gets per-domain β optimization (default: 3, or preset
    /// value)
    ///
    /// Deeper domains reuse the β values they inherit. Applies only when
    /// --beta-iterations is positive.
    #[arg(long)]
    pub(crate) beta_max_depth: Option<usize>,

    /// Learning rate for β optimization (default: 0.15, or preset value)
    #[arg(long)]
    pub(crate) lr_beta: Option<f32>,

    /// Use CROWN-IBP for intermediate bounds (default: on)
    ///
    /// Turned off automatically on graph models, for both input and ReLU
    /// splitting.
    #[arg(
        long,
        default_value_t = true,
        value_parser = clap::value_parser!(bool),
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    pub(crate) crown_ibp: bool,

    /// Domains processed per batch (default: 64, or preset value; 1 = one at a
    /// time)
    #[arg(long)]
    pub(crate) batch_size: Option<usize>,

    /// Disable parallel child domain creation (default: enabled)
    #[arg(long, default_value_t = false)]
    pub(crate) sequential_children: bool,

    /// Request experimental GCP-CROWN cutting planes.
    ///
    /// Certificate-bearing cut authority is currently quarantined, so this
    /// request is rejected during verifier configuration validation.
    #[arg(
        long,
        default_value_t = false,
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

    /// Minimum domain depth for cut generation (default: 2, or preset value).
    /// Deeper domains produce more specific cuts.
    #[arg(long)]
    pub(crate) min_cut_depth: Option<usize>,

    /// Request near-miss cut generation (quarantined; not proof-derived).
    #[arg(long, default_value_t = false)]
    pub(crate) enable_near_miss_cuts: bool,

    /// Margin for near-miss cut generation (default: 0.1, or preset value). A
    /// fraction of the threshold, or absolute when the threshold is 0.
    #[arg(long)]
    pub(crate) near_miss_margin: Option<f32>,

    /// Request proactive cut generation (quarantined; not proof-derived).
    #[arg(long, default_value_t = false)]
    pub(crate) proactive_cuts: bool,

    /// Maximum number of proactive cuts to generate (default: 100, or preset value)
    #[arg(long)]
    pub(crate) max_proactive_cuts: Option<usize>,

    /// Enable BICCOS constraint strengthening for verified-domain cuts.
    #[arg(long, default_value_t = false)]
    pub(crate) biccos_constraint_strengthening: bool,

    /// Drop ratio for BICCOS constraint strengthening (default: 0.5, or preset
    /// value): a quantile over influence scores.
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

    /// Objective neurons per layer to tighten with --clip-interm-domain
    /// (default: 3, or preset value)
    #[arg(long)]
    pub(crate) clip_interm_topk: Option<usize>,

    /// Apply clip_interm_domain during alpha-CROWN optimization.
    #[arg(long, default_value_t = false)]
    pub(crate) clip_in_alpha_crown: bool,

    /// Reserved pruning authority. Rejected with --clip-interm-domain until
    /// certificate-backed pruning is implemented.
    #[arg(long, default_value_t = false)]
    pub(crate) clip_interm_prune: bool,

    /// Reserved final-layer pruning extension; no effect while pruning is quarantined.
    #[arg(long, default_value_t = false)]
    pub(crate) clip_interm_use_final_layer: bool,

    /// Enable static intermediate bound transfer for batched domains.
    #[arg(long, default_value_t = false)]
    pub(crate) interm_transfer: bool,

    /// Search for counterexamples with a PGD attack (default: on)
    ///
    /// Every counterexample is re-checked before `sat` is reported, so PGD
    /// cannot change a verified result. Its time is capped at a fraction of
    /// the timeout. A preset's `attack.pgd_order: skip` overrides this flag;
    /// use --no-pgd-attack to turn PGD off.
    #[arg(
        long = "pgd-attack",
        num_args = 0..=1,
        require_equals = true,
        default_value_t = true,
        default_missing_value = "true",
        action = clap::ArgAction::Set
    )]
    pub(crate) pgd_attack: bool,

    /// Turn off the PGD attack (overrides --pgd-attack), for example for
    /// reproducible A/B runs
    #[arg(long = "no-pgd-attack", default_value_t = false, action = clap::ArgAction::SetTrue)]
    pub(crate) no_pgd_attack: bool,

    /// Number of PGD attack restarts (default: 100, or preset value)
    #[arg(long)]
    pub(crate) pgd_restarts: Option<usize>,

    /// Number of PGD gradient steps per restart (default: 50, or preset value)
    #[arg(long)]
    pub(crate) pgd_steps: Option<usize>,

    /// Compute backend: cpu or wgpu (default: the preset's general.device, else
    /// chosen by problem size)
    ///
    /// wgpu is used only if the adapter passes its qualification checks;
    /// otherwise ny uses the CPU and reports why.
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

    /// Use the batched GPU branch-and-bound engine (input and ReLU splitting)
    #[arg(long, default_value_t = false)]
    pub(crate) gpu_bab: bool,

    /// Recompute every GPU BaB backward pass from scratch instead of reusing the
    /// parent domain's linear bounds (for A/B benchmarking)
    #[arg(long, default_value_t = false)]
    pub(crate) no_la_warm_start: bool,

    /// Complete verifier: auto (default), bab, or mip
    ///
    /// `auto` runs branch-and-bound and escalates to an exact MIP solved by
    /// ay when the result is inconclusive and the network is MIP-encodable
    /// (sequential Linear+ReLU within a size cap). Escalation can only turn
    /// unknown or timeout into a decided verdict. MIP needs a build with the
    /// `mip` feature; without it, `auto` behaves like `bab` and `mip` falls
    /// back to branch-and-bound.
    #[arg(long, value_enum, default_value_t = CompleteVerifierArg::Auto)]
    pub(crate) complete_verifier: CompleteVerifierArg,

    /// MIP solver for --complete-verifier mip (only `ay` is available)
    ///
    /// Legacy preset values (highs, scip, gurobi) resolve to ay with a
    /// warning.
    #[arg(long, value_enum)]
    pub(crate) mip_solver: Option<MipSolverArg>,

    /// Proof profile: `proof` (this subcommand's default) or `competition`
    ///
    /// `proof` enables optional artifacts and runs downgrade-only audits at
    /// full rates; `competition` skips optional artifacts by default and samples
    /// the audits. Verdict certifiers, the cover gate and guards run identically
    /// in both profiles. Printed as
    /// `[proof-profile] mode=<m> source=<entry|cli>`.
    #[arg(long, value_parser = parse_proof_profile)]
    pub(crate) proof_profile: Option<ny_core::proof_profile::ProofProfile>,

    /// Deprecated spelling of `--proof-profile competition` (kept so existing
    /// scripts keep working; recorded as `source=cli`).
    #[arg(
        long,
        default_value_t = false,
        hide = true,
        conflicts_with = "proof_profile"
    )]
    pub(crate) competition_mode: bool,

    /// Refuse (`unknown`) any verdict the external certificate format cannot
    /// represent
    ///
    /// Also always attempts the certificate sidecar. Off by default in both
    /// profiles: without it a verdict the lanes certified at runtime is
    /// published, and a missing artifact is recorded as `artifact=none` rather
    /// than refused.
    #[arg(long, default_value_t = false)]
    pub(crate) require_certificate: bool,

    /// Disable optional certificate emission (--require-certificate takes precedence).
    #[arg(long, default_value_t = false, conflicts_with = "emit_certificate")]
    pub(crate) no_certificate: bool,

    /// Attempt proof certificate emission at this path, regardless of profile
    /// (default path: `<model-stem>.cert.json`)
    #[arg(long)]
    pub(crate) emit_certificate: Option<PathBuf>,

    /// Removed compatibility flag. Passing it is an error; unsound GPU bounds
    /// are never admitted to a verdict.
    #[arg(long, default_value_t = false, hide = true)]
    pub(crate) allow_unsound_gpu_crown: bool,
}

/// `on` / `off` for `ny proof-check --refine`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OnOff {
    /// Enabled.
    On,
    /// Disabled.
    Off,
}

// `ny proof-check` arguments (ENGINE_PROGRAM §1.7 B2.1). A plain comment, not a
// doc comment, so the `Commands::ProofCheck` doc is the command's only help text.
#[derive(clap::Args, Debug, Clone)]
pub(crate) struct ProofCheckArgs {
    /// Proof bundle to check (`<RESULTS_FILE>.proof.jsonl`, format
    /// ny-proof-bundle/v2)
    pub(crate) bundle: PathBuf,

    /// ONNX model (default: the path recorded in the bundle; its SHA-256 must
    /// match the bundle's)
    #[arg(long)]
    pub(crate) onnx: Option<PathBuf>,

    /// VNN-LIB property (default: the path recorded in the bundle; its SHA-256
    /// must match the bundle's)
    #[arg(long)]
    pub(crate) vnnlib: Option<PathBuf>,

    /// Where to write the JSON report (default: `<BUNDLE>.check.json`)
    #[arg(long)]
    pub(crate) report: Option<PathBuf>,

    /// Worker threads (default: Rayon configuration)
    #[arg(long)]
    pub(crate) threads: Option<usize>,

    /// Memory budget for back-substitution, in GiB
    #[arg(long, default_value_t = 8.0)]
    pub(crate) mem_gib: f64,

    /// Time budget in seconds
    ///
    /// Anything not confirmed when the budget runs out is reported as
    /// unconfirmed.
    #[arg(long, default_value_t = 3600.0)]
    pub(crate) budget_s: f64,

    /// Tighten intermediate bounds by back-substitution
    ///
    /// Applies to the root bounds of every input region and, when a region's
    /// recorded bounds do not confirm it, to bounds re-derived over that
    /// region alone.
    #[arg(long, value_enum, default_value_t = OnOff::On)]
    pub(crate) refine: OnOff,

    /// Points tried per unconfirmed region when searching for a
    /// counterexample
    #[arg(long, default_value_t = 16)]
    pub(crate) refute_points: usize,

    /// Print the full report as JSON on stdout
    #[arg(long)]
    pub(crate) json: bool,
}

// Boxing the largest argument groups bounds stack use in Clap's generated debug schema builder.
#[derive(Subcommand)]
pub(crate) enum Commands {
    /// Serve one bounded untrusted AY head-phase multiplier request
    #[cfg(feature = "mip")]
    #[command(name = "__head-phase-ay-advice", hide = true)]
    InternalHeadPhaseAyAdvice,

    /// Serve one bounded untrusted AY global head-functional request
    #[cfg(feature = "mip")]
    #[command(name = "__head-functional-ay-advice", hide = true)]
    InternalHeadFunctionalAyAdvice,

    /// Serve one bounded untrusted AY complete phase-cover batch
    #[cfg(feature = "mip")]
    #[command(name = "__head-phase-cover-ay-advice", hide = true)]
    InternalHeadPhaseCoverAyAdvice,

    /// Check a property using one bounding method (see --method)
    Verify(Box<VerifyArgs>),

    /// Certify a complete affine vocabulary head over a supplied hidden box
    HeadVerify {
        /// JSON request with original binary64 head paths, target and hidden box
        request: PathBuf,
        /// Independently replay a previously exported evidence file
        #[arg(long)]
        replay: Option<PathBuf>,
        /// Export replayable evidence to a new file
        #[arg(long)]
        evidence: Option<PathBuf>,
        /// Whole command deadline, including head loading and replay
        #[arg(long, default_value_t = 60)]
        timeout: u64,
        /// Original head allocation limit in MiB
        #[arg(long, default_value_t = 512)]
        max_head_mb: usize,
    },

    /// Show a model's structure
    Inspect {
        /// Path to model (ONNX, PyTorch, SafeTensors, CoreML, or GGUF)
        model: PathBuf,

        /// Load with the native importer (PyTorch, SafeTensors, CoreML, or GGUF)
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

    /// Show the trace-bridge operator soundness taxonomy
    ///
    /// Each TraceOp is classified as exact, sound, sound_but_loose or unsupported.
    /// This describes possible relaxations, not current importer coverage or the
    /// numeric soundness of a run. New variants must be classified to compile.
    Coverage {
        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Compute a certified global Lipschitz upper bound
    ///
    /// Uses exact rational arithmetic and fails closed outside the
    /// Linear/Conv/ReLU fragment. An optimistic spectral-norm estimate is shown
    /// alongside.
    Lipschitz {
        /// Path to ONNX model
        model: PathBuf,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Compare the propagated output bounds of two single-input ONNX models
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
        #[arg(long, default_value_t = false, hide = true, conflicts_with = "backend")]
        gpu: bool,

        /// Print per-element comparison details
        #[arg(long, default_value_t = false)]
        verbose: bool,

        /// Output as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Find where two ONNX models diverge, layer by layer
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

    /// Find which layers amplify input noise
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

    /// Check whether ONNX layers can be safely quantized to float16 or int8
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

    /// Show how bound width grows through the network
    ProfileBounds {
        /// Path to model (ONNX, PyTorch, SafeTensors, CoreML, or GGUF)
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

        /// Load with the native importer (PyTorch, SafeTensors, CoreML, or GGUF)
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

    /// Whisper verification (not available; always fails closed)
    Whisper {
        /// Retained model path argument; the unavailable command does not open it
        model: PathBuf,

        /// Retained component selector; ignored while verification is unavailable
        #[arg(short, long, default_value = "encoder")]
        component: String,

        /// Retained layer selector; ignored while verification is unavailable
        #[arg(short, long)]
        layer: Option<usize>,

        /// Retained perturbation value; ignored while verification is unavailable
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// Retained compatibility flag; the fail-closed error uses normal CLI text
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Unavailable experimental Whisper sequential command (fails closed)
    #[command(hide = true)]
    WhisperSeq {
        #[command(flatten)]
        common: WhisperCommonArgs,

        /// Retained perturbation value; ignored while verification is unavailable
        #[arg(short, long, default_value = "0.01")]
        epsilon: f32,

        /// Retained configuration preset; ignored while verification is unavailable
        #[arg(long, default_value = "default")]
        mode: String,

        /// Retained overflow option; ignored while verification is unavailable
        #[arg(long, value_parser = clap::value_parser!(bool))]
        terminate_on_overflow: Option<bool>,

        /// Retained overflow option; ignored while verification is unavailable
        #[arg(long, value_parser = clap::value_parser!(bool))]
        continue_after_overflow: Option<bool>,

        /// Retained clamp value; ignored while verification is unavailable
        #[arg(long)]
        overflow_clamp_value: Option<f32>,
    },

    /// Unavailable experimental Whisper epsilon-sweep command (fails closed)
    #[command(hide = true)]
    WhisperSweep {
        #[command(flatten)]
        common: WhisperCommonArgs,

        /// Retained sweep minimum; ignored while verification is unavailable
        #[arg(long, default_value = "0.000001")]
        epsilon_min: f32,

        /// Retained sweep maximum; ignored while verification is unavailable
        #[arg(long, default_value = "0.01")]
        epsilon_max: f32,

        /// Retained sweep size; ignored while verification is unavailable
        #[arg(long, default_value_t = 10)]
        steps: usize,

        /// Retained sweep-spacing flag; ignored while verification is unavailable
        #[arg(long, default_value_t = false)]
        linear: bool,

        /// Retained configuration preset; ignored while verification is unavailable
        #[arg(long, default_value = "strict")]
        mode: String,

        /// Retained reporting flag; ignored while verification is unavailable
        #[arg(long, default_value_t = false)]
        per_block: bool,
    },

    /// Unavailable experimental Whisper epsilon-search command (fails closed)
    #[command(hide = true)]
    WhisperEpsSearch {
        #[command(flatten)]
        common: WhisperCommonArgs,

        /// Retained target block count; ignored while verification is unavailable
        #[arg(long)]
        target_blocks: Option<usize>,

        /// Retained search minimum; ignored while verification is unavailable
        #[arg(long, default_value = "0.000001")]
        epsilon_min: f32,

        /// Retained search maximum; ignored while verification is unavailable
        #[arg(long, default_value = "0.01")]
        epsilon_max: f32,

        /// Retained iteration count; ignored while verification is unavailable
        #[arg(long, default_value_t = 20)]
        iterations: usize,

        /// Retained configuration preset; ignored while verification is unavailable
        #[arg(long, default_value = "strict")]
        mode: String,

        /// Retained progress flag; ignored while verification is unavailable
        #[arg(long, default_value_t = false)]
        verbose_search: bool,
    },

    /// Generate a Python script that exports a Whisper encoder to ONNX
    Export {
        /// Model type
        #[arg(short, long, default_value = "whisper", value_parser = ["whisper"])]
        model_type: String,

        /// Whisper model size
        #[arg(
            short,
            long,
            default_value = "tiny",
            value_parser = ["tiny", "base", "small", "medium", "large"]
        )]
        size: String,

        /// Output script path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Run built-in performance benchmarks (layer, attention, full, acasxu)
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

    /// Download, inventory, sweep, and score VNN-COMP corpora
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

    /// Prove or falsify a property with β-CROWN branch-and-bound
    BetaCrown(Box<BetaCrownArgs>),

    /// Run one VNN-COMP instance (v1) or print its run plan (plan)
    Vnncomp {
        #[command(subcommand)]
        action: VnncompAction,
    },

    /// Run explicit, non-verdict VNN-COMP research probes.
    #[command(hide = true)]
    VnncompResearch {
        #[command(subcommand)]
        action: commands::vnncomp::VnncompResearchAction,
    },

    /// Inspect and compare model weights (ONNX, SafeTensors, PyTorch, CoreML, or GGUF)
    Weights {
        #[command(subcommand)]
        action: commands::weights::WeightsAction,
    },

    /// Verify networks against geometric ground-truth specs (.gt.json)
    Gt {
        #[command(subcommand)]
        action: commands::gt::GtAction,
    },

    /// Check the proof bundle behind an `unsat` result, independently of the
    /// verifier
    ///
    /// Re-derives every bound the bundle claims with the checker's own ONNX
    /// and VNN-LIB readers and outward-rounded arithmetic, checks that the
    /// proved regions cover the whole input region for every disjunct, and
    /// writes a JSON report. It can only confirm or reject a proof, never
    /// produce or upgrade a result. Exit codes: 0 verified, 10 unconfirmed,
    /// 20 invalid, 21 refuted, 30 malformed, 31 unsupported, 40 checker fault,
    /// 4 usage or I/O error.
    ProofCheck(ProofCheckArgs),

    /// Re-check recorded empirical claims against their pinned artifacts
    ///
    /// Prints TRUE, FALSE, STALE, VACUOUS, or UNGROUNDED for each claim in the
    /// ledger (default: `claims/`). Each claim records the content-pinned
    /// artifacts it was derived from, how to re-derive its value, and the
    /// observables that show the measurement engaged. The same check ships as
    /// the standalone `ny-claims` binary, which builds much faster.
    Claims(ny_claims::cli::ClaimsArgs),

    /// Short verification lessons and a runnable demo (start with no argument)
    Tutorial {
        #[command(subcommand)]
        topic: Option<commands::tutorial::TutorialTopic>,
    },
}

/// Run the VNN-COMP protocol or inspect its resolved plan.
#[derive(Subcommand)]
pub(crate) enum VnncompAction {
    /// Run one VNN-COMP instance exactly as the competition harness does
    ///
    /// Loads the category preset, verifies with automatically chosen settings,
    /// and writes RESULTS_FILE: the first line is `unsat`, `sat`, `timeout`,
    /// `unknown`, or `error`, and for `sat` the counterexample follows.
    ///
    /// Protocol: `ny vnncomp v1 CATEGORY ONNX VNNLIB RESULTS_FILE TIMEOUT`.
    V1 {
        /// Benchmark category (e.g. `acasxu_2023`, `cifar100_2024`). Drives preset auto-loading.
        category: String,

        /// Path to the ONNX model for this instance.
        onnx: PathBuf,

        /// Path to the VNN-LIB property file for this instance.
        vnnlib: PathBuf,

        /// File to write the result to (plus the counterexample for `sat`).
        results_file: PathBuf,

        /// Time budget in seconds (fractional values such as `210.0` are floored).
        #[arg(value_parser = parse_budget_secs)]
        timeout_secs: u64,

        /// Directory holding the `vnncomp*/<CATEGORY>.yaml` presets (default:
        /// $NY_CONFIGS_DIR, else the nearest `configs/` above the binary, the ONNX
        /// file, or the working directory).
        #[arg(long)]
        configs_dir: Option<PathBuf>,

        /// Override the scored entry point's `competition` proof profile
        /// (#soundness-spine M2a) — `proof` for banking and moat runs. It
        /// selects artifact sinks and audit rates only; certification is
        /// identical in both profiles.
        #[arg(long, value_parser = parse_proof_profile)]
        proof_profile: Option<ny_core::proof_profile::ProofProfile>,

        /// Detail level of the per-phase cost census written to
        /// `<RESULTS_FILE>.flight.json` (`phase_cost`): `off`, `coarse`
        /// (default) or `detail`. Observability only; it never changes a
        /// verdict.
        #[arg(long, value_name = "LEVEL", value_parser = parse_phase_cost_level)]
        phase_cost: Option<ny_core::phase_cost::DetailLevel>,
    },

    /// Print the resolved run plan for one instance without running it
    ///
    /// Shows the detected host and backend, facts about the loaded model, each
    /// resolved setting with where it came from, and how the time budget is
    /// split.
    Plan {
        /// Benchmark category (e.g. `cifar100_2024`). Drives preset auto-loading.
        /// Required unless `--replay`.
        #[arg(required_unless_present = "replay")]
        category: Option<String>,

        /// Path to the ONNX model for this instance (facts come from the loaded
        /// model, never from its filename).
        #[arg(required_unless_present = "replay")]
        onnx: Option<PathBuf>,

        /// Path to the VNN-LIB property file for this instance.
        #[arg(required_unless_present = "replay")]
        vnnlib: Option<PathBuf>,

        /// Time budget in seconds (fractional values are floored).
        #[arg(value_parser = parse_budget_secs, required_unless_present = "replay")]
        budget_secs: Option<u64>,

        /// Directory holding the `vnncomp*/<CATEGORY>.yaml` presets (default:
        /// $NY_CONFIGS_DIR, else the nearest `configs/` above the binary, the ONNX
        /// file, or the working directory).
        #[arg(long)]
        configs_dir: Option<PathBuf>,

        /// Output as JSON.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// Also print the router's plan for this instance
        ///
        /// This is the plan a scored run logs when its preset sets
        /// `router.mode: shadow`.
        #[arg(long, default_value_t = false)]
        router: bool,

        /// Plan the router for another host: gb10, g5, m5, or other
        ///
        /// For measurement only. It changes the plan's values, never which lane
        /// runs.
        #[arg(long, value_name = "HOST")]
        router_target_host: Option<String>,

        /// Replay a logged router plan from a `<result>.flight.json` file
        ///
        /// Re-runs the router's plan and checkpoints on the inputs the flight
        /// file recorded, and exits non-zero on any divergence. Takes no
        /// instance arguments.
        #[arg(long, value_name = "FLIGHT_JSON")]
        replay: Option<PathBuf>,
    },

    /// Write the router's admission census over a corpus of instances
    ///
    /// For each corpus row, extracts the facts a scored run's router extracts
    /// and records, for every routable capability, whether the router would
    /// route it if it were sealed: the rows a seal must cover. Runs no
    /// verification. Prints one `CENSUS_DONE rc=0 ...` line.
    Census {
        /// The corpus manifest: a `# ny-admission-census-corpus/v1` line, then
        /// one `ROW_ID<TAB>ONNX<TAB>VNNLIB<TAB>BUDGET` line per row.
        #[arg(long, value_name = "CORPUS_TSV")]
        manifest: PathBuf,

        /// The census table to write (refused if it exists).
        #[arg(long, value_name = "CENSUS_TSV")]
        out: PathBuf,

        /// The seal envelope to assume: `widest` (every operator kind but
        /// `other`, every specification archetype) or `OPS/SPECS` with
        /// `+`-joined tokens (`conv+relu+linear/robustness`).
        #[arg(long, default_value = "widest")]
        envelope: String,

        /// Plan for another host: gb10, g5, m5, or other (default: this host)
        #[arg(long, value_name = "HOST")]
        router_target_host: Option<String>,

        /// Deadline of each parent-lane admission probe, in seconds
        #[arg(long, default_value_t = 60)]
        row_deadline_secs: u64,

        /// Where decompressed (competition-form) models and properties go
        /// (default: `<CENSUS_TSV>.work`)
        #[arg(long, value_name = "DIR")]
        work_dir: Option<PathBuf>,
    },
}

/// Parse a `--phase-cost` value: exactly `off`, `coarse` or `detail` (#phase-cost).
fn parse_phase_cost_level(raw: &str) -> Result<ny_core::phase_cost::DetailLevel, String> {
    ny_core::phase_cost::DetailLevel::parse(raw)
}

/// Parse a `--proof-profile` value: exactly `proof` or `competition`. A typo is
/// an error at the flag, never a silent fallback to the other profile.
fn parse_proof_profile(raw: &str) -> Result<ny_core::proof_profile::ProofProfile, String> {
    ny_core::proof_profile::ProofProfile::parse(raw)
        .ok_or_else(|| format!("expected `proof` or `competition`, got `{raw}`"))
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
