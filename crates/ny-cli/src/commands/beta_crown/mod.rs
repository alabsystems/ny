// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! β-CROWN branch-and-bound verification command handler.
//!
//! Provides CLI handler for `ny beta-crown` - complete neural network verification
//! using branch-and-bound with β-CROWN bound computation. Supports:
//! - VNN-LIB property files with relational and constant constraints
//! - NNet and ONNX model formats
//! - Sequential and DAG (GraphNetwork) models
//! - Input splitting and ReLU splitting for DAG models
//! - GCP-CROWN cutting planes
//! - PGD attack for counterexample search
//! - GPU acceleration

#[cfg(feature = "mip")]
mod ay_tail_authority;
pub(crate) mod best_margin_export;
pub(crate) mod branching;
mod build_batch_size;
mod cell_enum;
mod cert_adapter;
mod cnf_route;
pub(crate) mod constraint_eval;
pub(crate) mod constraint_plan;
mod dispatch;
mod domain_batch_metrics;
pub(crate) mod engine_dispatch;
mod frac_head;
#[cfg(feature = "mip")]
mod graph_mip;
#[cfg(feature = "mip")]
mod graph_mip_diff_coupling;
#[cfg(feature = "mip")]
mod graph_mip_escalate;
#[cfg(feature = "mip")]
mod graph_mip_fold;
#[cfg(feature = "mip")]
mod graph_mip_joint_relu_cuts;
#[cfg(feature = "mip")]
mod graph_mip_leaf;
#[cfg(feature = "mip")]
pub(crate) use graph_mip_leaf::relational_edge_milp_oracle;
#[cfg(feature = "mip")]
pub(crate) use graph_mip_leaf::whole_net_certified_band_unsat;
mod ibp_check;
mod input_split_metrics;
mod inputs;
mod invprop;
#[cfg(feature = "mip")]
mod mip_highs;
#[cfg(feature = "mip")]
mod mip_preprocess;
#[cfg(feature = "mip")]
mod mip_single_hidden;
mod model_load;
mod output;
mod routing;
#[cfg(not(feature = "mip"))]
mod mip_highs {
    use anyhow::Result;
    use ny_tensor::BoundedTensor;

    use super::BetaCrownModel;

    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify_with_mip(
        _model_net: &BetaCrownModel,
        _input: &BoundedTensor,
        _vnnlib: Option<&ny_onnx::vnnlib::VnnLibSpec>,
        _property: Option<&std::path::Path>,
        _epsilon: f32,
        _threshold: f32,
        _timeout: u64,
        _warm_start_candidate: Option<&ndarray::ArrayD<f32>>,
        _mip_solver: crate::MipSolverArg,
        _json: bool,
    ) -> Result<()> {
        anyhow::bail!(
            "MIP verification is disabled. Rebuild ny-cli with --features mip to enable."
        );
    }
}
mod verify;

// Re-export the in-process verdict-capture API so the native `vnncomp` subcommand
// can drive `handle_beta_crown_command` and read the rendered competition JSON
// without shelling out to a second `ny` process.
pub(crate) use output::{begin_capture, end_capture, take_captured_json};

use anyhow::Result;
use ny_gpu::ComputeDevice;
use ny_propagate::{BetaCrownConfig, BranchingHeuristic, VggMaxPoolRewriteMode};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

use crate::preset;

use self::dispatch::{run_bab_with_fallback, run_mip_only, DispatchContext};
use self::inputs::{check_needs_squeeze, create_input_bounds};
use self::invprop::{maybe_enable_invprop, maybe_set_alpha_output_constraints};
use self::model_load::{load_model, LoadedModel};
use self::routing::resolve_beta_crown_backend;
use crate::{
    AlphaGradientMethodArg, AlphaOptimizerArg, BackendArg, CompleteVerifierArg, MipSolverArg,
};
use build_batch_size::maybe_apply_build_batch_size_autotune_4354;

/// Model representation for β-CROWN verification.
enum BetaCrownModel {
    Sequential(Box<ny_propagate::Network>),
    Graph(Box<ny_propagate::GraphNetwork>),
}

impl BetaCrownModel {
    fn set_logsoftmax_sound_mode(&mut self, sound: bool) -> usize {
        match self {
            Self::Sequential(network) => network.set_logsoftmax_sound_mode(sound),
            Self::Graph(graph) => graph.set_logsoftmax_sound_mode(sound),
        }
    }

    fn set_softmax_sound_mode(&mut self, sound: bool) -> usize {
        match self {
            Self::Sequential(network) => network.set_softmax_sound_mode(sound),
            Self::Graph(graph) => graph.set_softmax_sound_mode(sound),
        }
    }

    fn set_causal_softmax_sound_mode(&mut self, sound: bool) -> usize {
        match self {
            Self::Sequential(network) => network.set_causal_softmax_sound_mode(sound),
            Self::Graph(graph) => graph.set_causal_softmax_sound_mode(sound),
        }
    }
}

fn apply_heuristic_sound_modes(
    network: &mut BetaCrownModel,
    allow_heuristic_logsoftmax: bool,
    allow_heuristic_softmax: bool,
    json: bool,
) {
    if allow_heuristic_logsoftmax {
        let modified = network.set_logsoftmax_sound_mode(false);
        if modified > 0 && !json {
            warn!(
                "LogSoftmax CROWN using heuristic sampling for {} nodes (not provably sound).",
                modified
            );
        }
    }

    if allow_heuristic_softmax {
        let modified_softmax = network.set_softmax_sound_mode(false);
        if modified_softmax > 0 && !json {
            warn!(
                "Softmax CROWN using heuristic sampling for {} nodes (not provably sound).",
                modified_softmax
            );
        }

        let modified_causal = network.set_causal_softmax_sound_mode(false);
        if modified_causal > 0 && !json {
            warn!(
                "CausalSoftmax CROWN using heuristic sampling for {} nodes (not provably sound).",
                modified_causal
            );
        }
    }
}

fn effective_interm_transfer(
    interm_transfer: bool,
    preset_config: Option<&preset::PresetConfig>,
) -> bool {
    if interm_transfer {
        true
    } else {
        preset_config
            .and_then(|p| p.bab.interm_transfer)
            .unwrap_or_else(|| BetaCrownConfig::default().enable_interm_transfer)
    }
}

/// What the preset's `attack.pgd_order` says about PGD enablement:
/// `Some(false)` for "skip"/"none"/"disabled", `Some(true)` for any other
/// order value, `None` when the preset does not speak.
fn preset_pgd_enabled(preset_config: Option<&preset::PresetConfig>) -> Option<bool> {
    preset_config?.attack.pgd_order.as_ref().map(|order| {
        !matches!(
            order.to_ascii_lowercase().as_str(),
            "skip" | "none" | "disabled"
        )
    })
}

fn preset_uses_input_bab_pgd_schedule(preset_config: Option<&preset::PresetConfig>) -> bool {
    preset_config.is_some_and(|preset| {
        preset
            .attack
            .pgd_order
            .as_ref()
            .is_some_and(|order| order.eq_ignore_ascii_case("input_bab"))
    })
}

/// Resolve PGD enablement from the CLI flag and the preset.
///
/// `--pgd-attack` defaults to true, so a `true` here cannot distinguish an
/// explicit request from the clap default; the preset's `attack.pgd_order` is
/// the more explicit signal and takes precedence (same shape as
/// [`effective_interm_transfer`], inverted for a default-ON flag).
/// `--no-pgd-attack` / `--pgd-attack=false` is always explicit and wins
/// outright. With no preset signal, PGD stays default-on.
fn resolve_pgd_attack(pgd_attack: bool, preset_config: Option<&preset::PresetConfig>) -> bool {
    if !pgd_attack {
        return false;
    }
    preset_pgd_enabled(preset_config).unwrap_or(true)
}

fn effective_upfront_pgd(
    enable_pgd_attack: bool,
    preset_config: Option<&preset::PresetConfig>,
) -> bool {
    enable_pgd_attack && !preset_uses_input_bab_pgd_schedule(preset_config)
}

#[derive(Debug, Clone, Copy)]
struct PropertyInputSummary {
    element_count: usize,
    perturbed_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VggAbcrownDecision {
    perturbed_count: usize,
    rewrite_mode: VggMaxPoolRewriteMode,
    use_forward_bounds: bool,
    prioritize_attack: bool,
}

fn count_perturbed_inputs(spec: &ny_onnx::vnnlib::VnnLibSpec) -> usize {
    spec.input_bounds
        .iter()
        .filter(|(lower, upper)| upper > lower)
        .count()
}

/// Resolve alpha-beta-CROWN's property-size VGG policy.
///
/// - <=100 perturbed scalars: forward+backward and the add-free rewrite.
/// - >100: plain CROWN and the residual Conv/ReLU/Add rewrite.
/// - >10,000: additionally move PGD from `input_bab` to the upfront stage.
///
/// The model field is the master gate. With it absent/false, or without an
/// authoritative VNN-LIB width count, the treatment is a complete no-op.
fn resolve_vgg_abcrown_decision(
    preset_config: Option<&preset::PresetConfig>,
    perturbed_count: Option<usize>,
) -> Option<VggAbcrownDecision> {
    if preset_config.is_none_or(|preset| preset.model.vgg_abcrown_treatment != Some(true)) {
        return None;
    }
    let perturbed_count = perturbed_count?;
    let use_forward_bounds = perturbed_count <= 100;
    Some(VggAbcrownDecision {
        perturbed_count,
        rewrite_mode: if use_forward_bounds {
            VggMaxPoolRewriteMode::Sequential
        } else {
            VggMaxPoolRewriteMode::Residual
        },
        use_forward_bounds,
        prioritize_attack: perturbed_count > 10_000,
    })
}

fn apply_vgg_abcrown_bound_mode(
    config: &mut BetaCrownConfig,
    decision: Option<VggAbcrownDecision>,
) {
    if let Some(decision) = decision {
        config.use_alpha_crown = false;
        config.use_forward_bounds = decision.use_forward_bounds;
    }
}

fn effective_upfront_pgd_with_vgg(
    enable_pgd_attack: bool,
    preset_config: Option<&preset::PresetConfig>,
    decision: Option<VggAbcrownDecision>,
) -> bool {
    enable_pgd_attack
        && (decision.is_some_and(|decision| decision.prioritize_attack)
            || effective_upfront_pgd(true, preset_config))
}

/// Peek the network input dimensions from the VNN-LIB property, if present.
///
/// Used to resolve `--branching auto` BEFORE the model is loaded (so DAG routing
/// is correct). The VNN-LIB spec's `num_inputs` equals the model's flattened input
/// element count (validated against `input_dim` in `create_input_bounds`), making
/// it a cheap, authoritative pre-load signal. Returns `None` when there is no
/// property file (epsilon-ball mode) or the spec fails to load; callers then defer
/// to the post-load resolution using the model's `input_dim`.
fn peek_property_input_summary(property: Option<&std::path::Path>) -> Option<PropertyInputSummary> {
    let prop_path = property?;
    match ny_onnx::vnnlib::load_vnnlib(prop_path) {
        Ok(spec) => Some(PropertyInputSummary {
            element_count: spec.num_inputs,
            perturbed_count: count_perturbed_inputs(&spec),
        }),
        Err(e) => {
            info!(
                "Could not peek input dimensions from {}: {} (deferring to post-load)",
                prop_path.display(),
                e
            );
            None
        }
    }
}

/// Proof-carrying / certificate-emission options, bundled into one struct so
/// the (already very large) positional argument list of
/// [`handle_beta_crown_command`] grows by one parameter, not three.
///
/// SOUNDNESS: none of these options can change the soundness of the sat/unsat
/// verdict. They only control *extra* machine-checkable artifacts and internal
/// self-checks emitted on top of a verdict. The float-soundness posture (f64
/// intermediates, directed rounding, conservative verdict translation) is
/// baseline and is NEVER gated by these options.
#[derive(Debug, Clone, Default)]
pub(crate) struct ProofOpts {
    /// Competition mode: maximise verify-rate within the wall-clock budget by
    /// turning OFF certificate emission and internal self-checks. Default
    /// `false` (proof features ON). The VNN-COMP scored entry point sets this
    /// `true`; interactive `ny beta-crown` keeps it `false`.
    pub competition_mode: bool,
    /// Explicit certificate-emission override. `None` means "auto" — emit a
    /// certificate iff NOT in competition mode. `Some(true)` / `Some(false)`
    /// force emission on / off regardless of `competition_mode`.
    pub emit_certificate: Option<bool>,
    /// Destination for the certificate sidecar. `None` uses the default path
    /// (`<model>.cert.json` alongside the model).
    pub certificate_path: Option<PathBuf>,
    /// Speed override: when `true`, ALLOW the UNSOUND fast f32 GPU CROWN backward
    /// to decide verdicts (disengages the soundness gate). Default `false` — the
    /// gate is ENGAGED so every verdict is decided by a proven-sound path (sound
    /// GPU-resident backward or CPU f64+γ_n·S). Ignored under `competition_mode`
    /// (which forces sound). No effect on CPU-only runs.
    pub allow_unsound_gpu_crown: bool,
}

impl ProofOpts {
    /// Whether a proof-carrying certificate should be emitted for a `Verified`
    /// verdict, resolving the `emit_certificate` override against
    /// `competition_mode`.
    pub(crate) fn should_emit_certificate(&self) -> bool {
        self.emit_certificate.unwrap_or(!self.competition_mode)
    }

    /// Whether the process-global soundness gate should be ENGAGED — i.e. every
    /// verdict-deciding CROWN bound must take a proven-sound path (the unsound fast
    /// GPU f32 backward is masked). DEFAULT SOUND: a verifier never decides a verdict
    /// on an unsound bound unless the user KNOWINGLY opts into speed via
    /// `--allow-unsound-gpu-crown`, and `--competition-mode` can never opt out.
    pub(crate) fn sound_gpu_crown_required(&self) -> bool {
        self.competition_mode || !self.allow_unsound_gpu_crown
    }
}

/// Per-instance, typed overrides owned by the in-process VNN-COMP router.
/// Interactive CLI calls always pass `Default::default()`. Keeping this typed
/// avoids process-global environment mutation after an exact instance policy
/// has already been resolved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BetaCrownInstanceOverrides {
    /// Arm bounded sparse root CROWN for the sealed adaptive-release route.
    pub(crate) root_sparse_interm_crown: bool,
}

fn apply_instance_overrides(
    config: &mut BetaCrownConfig,
    instance_overrides: BetaCrownInstanceOverrides,
) {
    if instance_overrides.root_sparse_interm_crown {
        // Exact per-instance performance route. The root policy's raw
        // `NY_ROOT_SPARSE_INTERM_CROWN=0` resolver remains the final kill
        // switch and can still disable this typed-on config.
        config.root_sparse_interm_crown = true;
    }
}

/// Resolve the dark post-C survivor experiment exactly once at the CLI/config
/// boundary. Only the literal value `1` arms it; unset, malformed, and every
/// other value preserve the default-off policy. The propagation engine consumes
/// only the typed field and never consults process-global environment state.
fn root_post_c_survivor_enabled_from_value(value: Option<&str>) -> bool {
    value == Some("1")
}

fn apply_root_post_c_survivor_env(config: &mut BetaCrownConfig) {
    config.root_post_c_survivor = root_post_c_survivor_enabled_from_value(
        std::env::var("NY_ROOT_POST_C_SURVIVOR").ok().as_deref(),
    );
}

/// Handle the `ny beta-crown` command.
///
/// This is the complete verification entry point using branch-and-bound with:
/// - β-CROWN bound computation with α optimization
/// - Multiple branching heuristics (width, impact/BaBSR, FSB, input split, ReLU split)
/// - GCP-CROWN cutting planes for tighter bounds
/// - PGD attack for counterexample search
/// - Support for VNN-LIB property files with complex constraints
/// - Preset configuration files for benchmark-specific tuning
// Justification: Top-level CLI command handler — parameters map 1:1 to clap arguments
// that users pass on the command line. A config struct would duplicate SubCommands enum.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_beta_crown_command(
    model: PathBuf,
    property: Option<PathBuf>,
    preset: Option<PathBuf>,
    epsilon: f32,
    threshold: f32,
    peel_last_softmax_layer: bool,
    allow_heuristic_logsoftmax: bool,
    allow_heuristic_softmax: bool,
    max_domains: Option<usize>,
    timeout: Option<u64>,
    max_depth: Option<usize>,
    branching: Option<String>,
    fsb_candidates: Option<usize>,
    no_alpha: bool,
    alpha_iterations: Option<usize>,
    input_split_alpha_iterations: Option<usize>,
    input_split_lr_alpha: Option<f32>,
    no_adaptive_alpha_skip: bool,
    alpha_skip_depth: Option<usize>,
    crown_ibp_intermediates: bool,
    alpha_spsa_samples: Option<usize>,
    alpha_lr: Option<f32>,
    alpha_gradient_method: Option<AlphaGradientMethodArg>,
    alpha_optimizer: Option<AlphaOptimizerArg>,
    invprop: bool,
    invprop_apply: Vec<String>,
    invprop_share_gammas: bool,
    beta_iterations: Option<usize>,
    beta_max_depth: Option<usize>,
    lr_beta: Option<f32>,
    crown_ibp: bool,
    batch_size: Option<usize>,
    sequential_children: bool,
    enable_cuts: bool,
    no_cuts: bool,
    max_cuts: Option<usize>,
    min_cut_depth: Option<usize>,
    enable_near_miss_cuts: bool,
    near_miss_margin: Option<f32>,
    proactive_cuts: bool,
    max_proactive_cuts: Option<usize>,
    biccos_constraint_strengthening: bool,
    biccos_drop_ratio: Option<f32>,
    relaxed_clip: bool,
    relaxed_clip_iterations: Option<usize>,
    clip_interm_domain: bool,
    clip_interm_topk: Option<usize>,
    clip_in_alpha_crown: bool,
    clip_interm_prune: bool,
    clip_interm_use_final_layer: bool,
    interm_transfer: bool,
    pgd_attack: bool,
    pgd_restarts: Option<usize>,
    pgd_steps: Option<usize>,
    backend: Option<BackendArg>,
    gpu: bool,
    // Capability hint for the AUTO size gate: whether a usable GPU backend is
    // present (compiled in AND, for the scored VNN-COMP path, runtime-probed via
    // `GPU_AVAILABLE`). This is NOT a force — unlike `gpu`/`--backend`/preset, it
    // only tips the AUTO default toward the GPU for LARGE conv-dominated inputs;
    // small input-split nets (ACAS) still route to CPU regardless. `None` =>
    // fall back to the compile-time `wgpu_backend_compiled()` default.
    gpu_available: Option<bool>,
    input_split_metrics_jsonl: Option<PathBuf>,
    domain_batch_metrics_jsonl: Option<PathBuf>,
    json: bool,
    gpu_bab: bool,
    no_la_warm_start: bool,
    complete_verifier: CompleteVerifierArg,
    mip_solver: Option<MipSolverArg>,
    proof_opts: ProofOpts,
    instance_overrides: BetaCrownInstanceOverrides,
) -> Result<()> {
    #[cfg(feature = "mip")]
    graph_mip::install_imb_ay_tail_certificate_oracle();

    // Resolve effective backend from CLI flags and preset.
    // Precedence: --backend > --gpu > preset general.device > CPU default.
    let enable_cuts = enable_cuts && !no_cuts;

    // Load preset configuration if provided
    // Preset values establish baseline; CLI flags override them
    let preset_config = if let Some(ref preset_path) = preset {
        let loaded = preset::load_preset(preset_path)?;
        info!("Loaded preset config: {}", preset_path.display());
        Some(loaded)
    } else {
        None
    };
    let onnx_load_config = preset_config
        .as_ref()
        .map(preset::build_onnx_load_config)
        .transpose()?
        .unwrap_or_default()
        // Crash-isolate ORT shape inference (see `cli_shape_infer_backend`):
        // an ORT abort must cost at most the inferred shapes, never the
        // verifier process (and with it the chance to emit any verdict).
        .with_shape_infer_backend(crate::commands::cli_shape_infer_backend());

    let preset_device = preset_config
        .as_ref()
        .and_then(|p| p.general.device.as_deref());
    // Peek the model's flattened input element count from the VNN-LIB spec ONCE.
    // Reused for (a) the auto-backend default below and (b) auto-branching later
    // (see `auto_select_branching`). The spec's `num_inputs` equals the model
    // `input_dim` (validated in `create_input_bounds`), making it an authoritative
    // pre-load size signal. `None` in epsilon-ball mode (no spec).
    let property_input_summary = peek_property_input_summary(property.as_deref());
    let peeked_input_count = property_input_summary.map(|summary| summary.element_count);
    let vgg_abcrown_decision = resolve_vgg_abcrown_decision(
        preset_config.as_ref(),
        property_input_summary.map(|summary| summary.perturbed_count),
    );
    if preset_config
        .as_ref()
        .is_some_and(|preset| preset.model.vgg_abcrown_treatment == Some(true))
    {
        if let Some(decision) = vgg_abcrown_decision {
            info!(
                "VGG alpha-beta-CROWN treatment: {} perturbed inputs, mode={:?}, upfront_pgd={}",
                decision.perturbed_count, decision.rewrite_mode, decision.prioritize_attack
            );
        } else {
            warn!(
                "VGG alpha-beta-CROWN treatment requested without readable VNN-LIB bounds; \
                 leaving the model and dynamic policy unchanged"
            );
        }
    }
    // Register the instance model for lazily-built ORT-routed attack candidate
    // scoring (#four-walls). Session construction is deferred to the first
    // attack forward; non-attack runs pay nothing. NY_ORT_ATTACK=0 disables.
    verify::ort_attack::register_attack_model(model.clone(), peeked_input_count);
    // Auto-backend default: GPU for large conv-dominated inputs when a GPU is
    // available, else CPU. Explicit --backend / legacy --gpu / preset
    // general.device still take precedence (see resolve_beta_crown_backend).
    // SOUND: the backend is numerically faithful, so this only affects speed.
    // The capability hint defaults to compile-time wgpu availability; the scored
    // VNN-COMP path narrows it with the `GPU_AVAILABLE` runtime probe so we never
    // even *prefer* GPU on a box that has none.
    let gpu_available = gpu_available.unwrap_or_else(ny_gpu::wgpu_backend_compiled);
    let (effective_backend, auto_backend_reason) = resolve_beta_crown_backend(
        backend,
        gpu,
        preset_device,
        peeked_input_count,
        gpu_available,
    );
    if let Some(reason) = auto_backend_reason {
        info!(
            "Auto-backend: selected {} ({}); input_elements={:?}",
            effective_backend, reason, peeked_input_count
        );
        if !json {
            println!(
                "Auto-backend: selected {} ({}); input_elements={}",
                effective_backend,
                reason,
                peeked_input_count.map_or_else(|| "unknown".to_string(), |c| c.to_string())
            );
        }
    } else if backend.is_none() && !gpu && effective_backend != BackendArg::Cpu {
        info!("Preset selected {} backend", effective_backend);
    }
    let use_gpu_backend = effective_backend != BackendArg::Cpu;

    // `--no-alpha` remains the highest-precedence override. Otherwise a preset can
    // request plain CROWN via `solver.bound_prop_method: crown`, and that choice
    // must affect early routing/debug output before config assembly.
    let preset_use_alpha = preset_config
        .as_ref()
        .map(|preset| {
            preset::resolve_use_alpha_from_bound_prop_method(
                preset.solver.bound_prop_method.as_deref(),
            )
        })
        .transpose()?
        .flatten();
    let use_alpha = if no_alpha {
        false
    } else {
        preset_use_alpha.unwrap_or(true)
    };

    // GPU batching is now automatic for DAG models (Issue #12 implemented)
    // No warning needed - tensor-level batching reduces kernel launch overhead

    info!("Running β-CROWN verification on: {}", model.display());

    if peel_last_softmax_layer && property.is_none() && !json {
        eprintln!(
            "Warning: --peel-off-last-softmax-layer requires --property (VNN-LIB); flag ignored."
        );
    }

    // Parse branching heuristic (if CLI provided)
    // "relu" triggers ReLU-splitting for DAG models; None / "auto" defer to
    // preset, then to model-intrinsic auto-selection (see below).
    let cli_branching_auto = branching::is_auto_branching(branching.as_deref());
    // An explicit non-auto CLI token still wins over the preset (backward compat).
    // `auto` (the default) is treated like "not provided" so a preset's
    // `bab.branching.method` keeps control, exactly as the VNN-COMP runner did
    // when it dropped its own `--branching` flag for preset-owned categories.
    let cli_branching_provided = branching.is_some() && !cli_branching_auto;
    let (mut cli_branching_heuristic, cli_use_relu_split) =
        branching::parse_branching_with_relu(branching.as_deref())?;
    let preset_branching = preset_config
        .as_ref()
        .map(preset::resolve_branching)
        .transpose()?
        .flatten();
    let preset_owns_branching = preset_branching.is_some();
    let mut effective_branching_heuristic = if cli_branching_provided {
        cli_branching_heuristic.clone()
    } else {
        preset_branching
            .as_ref()
            .map(|resolved| resolved.heuristic.clone())
    };
    let mut use_relu_split = if cli_branching_provided {
        cli_use_relu_split
    } else {
        preset_branching
            .as_ref()
            .is_some_and(|resolved| resolved.use_relu_split)
    };

    // complete_verifier: an explicit CLI `--complete-verifier bab|mip` wins;
    // otherwise a preset's `general.complete_verifier` (sat_relu/malbeware route
    // to MIP with the full budget instead of burning it in BaB first); the clap
    // default `auto` preserves the preset value, mirroring the boolean-flag rule.
    let complete_verifier = match (
        complete_verifier,
        resolve_preset_complete_verifier(preset_config.as_ref())?,
    ) {
        (CompleteVerifierArg::Auto, Some(from_preset)) => from_preset,
        (cli, _) => cli,
    };

    // mip_solver: ay is the only backend (SOLVER POLICY, ny-mip
    // docs/SOLVER_POLICY.md); the preset resolver maps legacy foreign names
    // to ay with a warning.
    let mip_solver = mip_solver
        .or(resolve_preset_mip_solver(preset_config.as_ref())?)
        .unwrap_or(MipSolverArg::AY);

    // Whether the active complete verifier is MIP. `--complete-verifier mip`
    // routes the real work through the exact MIP solver, so auto-branching prefers
    // ReLU/kFSB for the BaB fallback (input splitting is hopeless on SAT-encoded /
    // NLP / malware nets).
    let mip_complete_verifier = complete_verifier == CompleteVerifierArg::Mip;

    // Model-CLASS-aware auto-branching is resolved INSIDE `load_model`, once the
    // network's structural signals (param_count, conv presence, ReLU-node count)
    // and the DAG flag are known — pure input dimensionality cannot separate the
    // dist_shift class (792-dim small conv autoencoder → input split) from
    // collins_rul (400-dim CNN → ReLU split). We build the request whenever the
    // user did not PIN an explicit branching method and no preset owns it; this
    // covers both `--branching auto` (the clap default the `beta-crown` CLI gets)
    // AND `branching == None` (what the `ny vnncomp` entry point passes directly to
    // the handler). Gating on `!cli_branching_provided` rather than `cli_branching_auto`
    // is the difference: `is_auto_branching(None)` is false, so the old
    // `cli_branching_auto` gate silently DROPPED auto-selection for the entire
    // `ny vnncomp` path — every DAG model (ResNet/ViT: tinyimagenet, cifar100,
    // vggnet, …) then hit the "Model is a DAG" bail and scored `unknown`, even
    // though the `beta-crown` CLI handled them fine. `load_model` returns the
    // resolved decision. The spec-peeked input count is passed only as a fallback
    // for the model `input_dim`. SOUND: the branching choice never changes a verdict.
    let mut auto_input_split_selected = false;
    let auto_request = (!cli_branching_provided && !preset_owns_branching).then_some(
        branching::AutoBranchingRequest {
            mip_complete_verifier,
            spec_input_count: peeked_input_count,
        },
    );

    // Parallel children is enabled by default, --sequential-children disables it
    let parallel_children = !sequential_children;

    // Load model (NNet or ONNX)
    let preset_conv_mode = preset_config
        .as_ref()
        .and_then(|preset| preset.general.conv_mode);
    let LoadedModel {
        model: mut model_net,
        param_count,
        input_dim,
        output_dim,
        input_shape: input_shape_from_model,
        is_graph: is_graph_model,
        mut preloaded_vnnlib,
        sigmoid_peeled,
        auto_branching,
    } = {
        let (loaded_model, routed_relu_split) = load_model(
            &model,
            &onnx_load_config,
            property.as_deref(),
            peel_last_softmax_layer,
            effective_branching_heuristic.as_ref(),
            use_relu_split,
            use_alpha,
            preset_conv_mode,
            enable_cuts,
            complete_verifier,
            json,
            auto_request,
            vgg_abcrown_decision.map(|decision| decision.rewrite_mode),
        )?;
        use_relu_split = routed_relu_split;
        loaded_model
    };

    // Consume the model-class-aware auto-branching decision (if `auto` was
    // requested and no preset owned branching). `load_model` already used it for
    // its own conv/DAG routing; here we mirror it into the config-facing state.
    if let Some(resolved) = auto_branching {
        auto_input_split_selected = resolved.is_input_split;
        if !json {
            println!(
                "Auto-branching: selected {:?} ({}); input_elements={}",
                resolved.heuristic,
                resolved.reason.as_str(),
                resolved.input_element_count
            );
        }
        info!(
            "Auto-branching selected {:?}: {} (input_elements={}, mip_complete={})",
            resolved.heuristic,
            resolved.reason.as_str(),
            resolved.input_element_count,
            mip_complete_verifier
        );
        // `load_model` already folded the resolved ReLU-split flag into
        // `routed_relu_split` (assigned to `use_relu_split` above); do not clobber
        // that with the pre-routing value.
        effective_branching_heuristic = Some(resolved.heuristic.clone());
        cli_branching_heuristic = Some(resolved.heuristic);
    }

    apply_heuristic_sound_modes(
        &mut model_net,
        allow_heuristic_logsoftmax,
        allow_heuristic_softmax,
        json,
    );

    match &model_net {
        BetaCrownModel::Sequential(network) => {
            if network.layers().is_empty() {
                anyhow::bail!("No layers in network - β-CROWN requires at least one layer");
            }
        }
        BetaCrownModel::Graph(graph) => {
            if graph.num_nodes() == 0 {
                anyhow::bail!("No nodes in graph - β-CROWN requires at least one operation");
            }
        }
    }

    // Initialize compute device if not CPU.
    // Wrap in Arc so the engine can be stored on BetaCrownVerifier structs (#3627).
    let compute_device: Option<Arc<ComputeDevice>> = if effective_backend != BackendArg::Cpu {
        match ComputeDevice::new(effective_backend.into()) {
            Ok(d) => {
                info!(
                    "Using {} backend for GPU-accelerated CROWN",
                    effective_backend
                );
                Some(Arc::new(d))
            }
            Err(e) => {
                eprintln!(
                    "Warning: Failed to create {} device: {}. Falling back to CPU.",
                    effective_backend, e
                );
                None
            }
        }
    } else if std::env::var("NY_ROOT_JOINT_INTERM_ALPHA").ok().as_deref() == Some("1") {
        // DARK (#root-joint-interm-alpha): the gated root-joint seam (and its
        // frozen-stop / image-node-crown machinery) needs a GPU engine even on
        // CPU-routed small-input models — cgan's 5-dim latent routes the auto
        // backend to CPU, leaving `engine=None` and no sound-GPU backward for
        // the seam. Create a wgpu device on demand INSTEAD of requiring a
        // preset `general.device` pin. Gate-off (the default) is byte-identical
        // to the shipped `None` arm; creation failure fails open to engine-less.
        match ComputeDevice::new(BackendArg::Wgpu.into()) {
            Ok(d) => {
                eprintln!(
                    "[root-joint-interm-alpha] on-demand wgpu engine created \
                     (CPU-routed model, gate on)"
                );
                Some(Arc::new(d))
            }
            Err(e) => {
                eprintln!(
                    "[root-joint-interm-alpha] on-demand wgpu engine unavailable: {e}; \
                     seam runs engine-less"
                );
                None
            }
        }
    } else {
        None
    };
    if use_gpu_backend && compute_device.is_none() {
        warn!("GPU backend requested but unavailable - using CPU");
    }

    // SOUNDNESS GATE (#vnncomp-gpu-crown-soundness). The *unsound fast* GPU CROWN
    // backward / concretize path computes bounds in round-to-nearest f32 with NO
    // γ_n·S certified rounding-error term, so such a bound can be tighter than the
    // true range and flip a genuinely-violated instance to a Verified/unsat verdict
    // (one wrong VNN-COMP verdict scores -150).
    //
    // DEFAULT: SOUND (#gpu-crown-sound-default, 2026-07-05). The gate is ENGAGED for
    // EVERY `ny beta-crown` / `ny verify` run — a VERIFIER must not decide a verdict
    // on an unsound bound by default. This is cheap: the gate does NOT force CPU (see
    // below), it routes to the *sound GPU-resident* backward, still GPU-accelerated.
    // A user who KNOWINGLY wants the faster unsound f32 path passes
    // `--allow-unsound-gpu-crown`; `--competition-mode` always forces sound (the
    // scored path can never opt out). CPU-only runs are unaffected (the CPU CROWN is
    // f64+γ_n·S regardless).
    //
    // NOTE (routing): the gate does NOT force CPU. Verdict sites call
    // `gpu_crown_backward_route`, which under the gate returns the *sound
    // GPU-resident* backward (`crown_backward_gpu_sound`, wgpu/Metal/CUDA engines
    // advertising `provides_sound_gpu_crown()`) when available — carrying verdicts at
    // GPU speed with directed/over-bounded error throughout — and only falls back to
    // the CPU f64 + γ_n·S path on `Err`/NaN or when no sound GPU backward exists.
    // What the gate *masks* is exclusively the unsound fast GPU f32 backward. GPU
    // GEMM, GPU IBP forward, and the PGD/attack (sat-finding) path keep their
    // acceleration regardless (PGD only exhibits a concrete, re-checked
    // counterexample, so its float speed can never produce an unsound Verified).
    ny_propagate::set_sound_gpu_crown_required(proof_opts.sound_gpu_crown_required());

    // Create GemmEngine reference for GPU-accelerated CROWN operations
    let gemm_engine = compute_device
        .as_deref()
        .map(|d| d as &dyn ny_core::GemmEngine);

    // Determine whether we need to squeeze batch dimension for Conv inputs.
    // This handles both:
    // 1. Direct Conv2d first layer (NCHW format)
    // 2. Transpose -> Conv2d (NHWC format converted to NCHW)
    let needs_squeeze = check_needs_squeeze(&model_net)?;

    // Create input bounds (VNNLIB or epsilon-ball)
    let (input, effective_threshold, vnnlib_spec, verify_upper, has_relational, const_output_idx) =
        create_input_bounds(
            &property,
            preloaded_vnnlib.take(),
            input_dim,
            output_dim,
            &input_shape_from_model,
            needs_squeeze,
            is_graph_model,
            epsilon,
            threshold,
            json,
        )?;

    // Softmax "complex" rewrite (W2 Phase B, vit_2023): decompose Softmax
    // nodes into the alpha-optimizable Exp/ReduceSum/Reciprocal/MulBinary
    // subgraph at graph-construction level, gated by the preset field
    // `solver.alpha-crown.softmax: complex` and the NY_NO_SOFTMAX_COMPLEX=1
    // kill-switch. Runs AFTER `create_input_bounds` because the shift
    // constant (rowwise max of the softmax input's interval center) needs
    // this instance's input box; `input` here is the same tensor every
    // subsequent propagation uses, so the frozen constants are consistent.
    // Ineligible nodes are skipped inside the rewrite (kept on the direct-LSE
    // path); an error leaves the graph unchanged — never a new failure mode.
    if resolve_preset_softmax_complex(preset_config.as_ref()) {
        if let BetaCrownModel::Graph(ref mut graph) = model_net {
            match graph.decompose_softmax_complex(&input) {
                Ok(report) => {
                    if !report.decomposed.is_empty() || !report.skipped.is_empty() {
                        info!(
                            "softmax-complex rewrite: {} decomposed, {} kept on direct-LSE path",
                            report.decomposed.len(),
                            report.skipped.len()
                        );
                        if !json && !report.decomposed.is_empty() {
                            println!(
                                "Softmax complex: decomposed {} softmax node(s) into alpha-optimizable primitives",
                                report.decomposed.len()
                            );
                        }
                    }
                }
                Err(e) => warn!(
                    "softmax-complex rewrite failed ({}); continuing with the current graph",
                    e
                ),
            }
        }
    }

    // For graph models, automatically disable features not yet supported
    // ReLU splitting now supports cuts (GCP-CROWN for DAGs)
    // Input splitting still requires disabling cuts
    let is_input_split_for_graph = effective_branching_heuristic
        .as_ref()
        .map(|h| matches!(h, BranchingHeuristic::InputSplit))
        .unwrap_or(false);
    let (use_alpha_effective, crown_ibp_effective, cuts_supported) = if is_graph_model
        && is_input_split_for_graph
    {
        // Input splitting: α-CROWN supported (tighter initial bounds via CROWN-IBP
        // intermediates in collect_alpha_crown_bounds_dag, #3357); crown_ibp and cuts
        // still not supported for input splitting.
        if (crown_ibp || enable_cuts) && !json {
            eprintln!(
                    "Note: Graph model with input splitting - disabling unsupported features (crown_ibp={}, cuts={})",
                    crown_ibp, enable_cuts
                );
        }
        (use_alpha, false, false)
    } else if is_graph_model && use_relu_split {
        // ReLU splitting: α-CROWN and cuts supported; crown_ibp still not supported
        if crown_ibp && !json {
            eprintln!(
                "Note: Graph model with ReLU splitting - disabling CROWN-IBP (crown_ibp={})",
                crown_ibp
            );
        }
        (use_alpha, false, true) // α-CROWN and cuts supported for graph ReLU splitting
    } else {
        (use_alpha, crown_ibp, true)
    };

    if !json {
        println!("Model: {}", model.display());
        if let Some(prop_path) = &property {
            println!("Property: {}", prop_path.display());
            if let Some(ref spec) = vnnlib_spec {
                println!("Input region: {} dimensions", spec.num_inputs);
                for (i, (l, u)) in spec.input_bounds.iter().enumerate() {
                    println!("  X_{}: [{:.6}, {:.6}] (width: {:.6})", i, l, u, u - l);
                }
            }
        } else {
            println!("Input shape: {:?}, epsilon: {}", input.shape(), epsilon);
        }
        let verify_msg = if verify_upper {
            format!(
                "output < {} (unsafe if output >= {})",
                effective_threshold, effective_threshold
            )
        } else {
            format!(
                "output > {} (unsafe if output <= {})",
                effective_threshold, effective_threshold
            )
        };
        println!(
            "Threshold: {} (verifying {})",
            effective_threshold, verify_msg
        );
        // Compute effective boolean values: CLI true wins, else preset value, else default (false).
        // This matches the config construction logic below.
        let eff_relaxed_clip = relaxed_clip
            || preset_config
                .as_ref()
                .and_then(|p| p.bab.clip.relaxed)
                .unwrap_or(false);
        let eff_pgd_attack = resolve_pgd_attack(pgd_attack, preset_config.as_ref());
        let eff_enable_cuts = resolve_enable_cuts(
            preset_config
                .as_ref()
                .and_then(|p| p.bab.cuts.enabled)
                .unwrap_or(false),
            enable_cuts,
            no_cuts,
            cuts_supported,
        );
        let eff_clip_interm_domain = clip_interm_domain
            || preset_config
                .as_ref()
                .and_then(|p| p.bab.clip.interm_domain)
                .unwrap_or(false);
        // `interm_transfer` falls back to the BetaCrownConfig default, not `false`.
        let eff_interm_transfer =
            effective_interm_transfer(interm_transfer, preset_config.as_ref());

        // Display effective values (from CLI, preset, or defaults)
        let branching_str = branching.as_deref().unwrap_or("(preset/default)");
        println!(
            "Config: max_domains={}, timeout={}s, max_depth={}, branching={}, fsb_candidates={}, use_alpha={}, alpha_iter={}, alpha_grad={}, alpha_opt={}, alpha_lr={}, crown_ibp_interm={}, beta_iter={}, lr_beta={}, crown_ibp={}, batch_size={}, parallel={}, enable_cuts={}, max_cuts={}, min_cut_depth={}, biccos_strengthen={}, biccos_drop_ratio={}, relaxed_clip={}, relaxed_clip_iters={}, clip_interm_domain={}, clip_interm_topk={}, clip_in_alpha_crown={}, clip_interm_prune={}, clip_interm_final_layer={}, interm_transfer={}, pgd_attack={}",
            max_domains.map(|v| v.to_string()).unwrap_or_else(|| "(preset/default)".to_string()),
            timeout.map(|v| v.to_string()).unwrap_or_else(|| "(preset/default)".to_string()),
            max_depth.map(|v| v.to_string()).unwrap_or_else(|| "(preset/default)".to_string()),
            branching_str,
            fsb_candidates.map(|v| v.to_string()).unwrap_or_else(|| "(preset/default)".to_string()),
            use_alpha_effective,
            alpha_iterations.map(|v| v.to_string()).unwrap_or_else(|| "(preset/default)".to_string()),
            alpha_gradient_method.map(|v| format!("{:?}", v)).unwrap_or_else(|| "(preset/default)".to_string()),
            alpha_optimizer.map(|v| format!("{:?}", v)).unwrap_or_else(|| "(preset/default)".to_string()),
            alpha_lr.map(|v| format!("{}", v)).unwrap_or_else(|| "(preset/default)".to_string()),
            crown_ibp_intermediates,
            beta_iterations.map(|v| v.to_string()).unwrap_or_else(|| "(preset/default)".to_string()),
            lr_beta.map(|v| format!("{}", v)).unwrap_or_else(|| "(preset/default)".to_string()),
            crown_ibp_effective,
            batch_size.map(|v| v.to_string()).unwrap_or_else(|| "(preset/default)".to_string()),
            parallel_children,
            eff_enable_cuts,
            max_cuts.map(|v| v.to_string()).unwrap_or_else(|| "(preset/default)".to_string()),
            min_cut_depth.map(|v| v.to_string()).unwrap_or_else(|| "(preset/default)".to_string()),
            biccos_constraint_strengthening,
            biccos_drop_ratio.map(|v| format!("{}", v)).unwrap_or_else(|| "(preset/default)".to_string()),
            eff_relaxed_clip,
            relaxed_clip_iterations.map(|v| v.to_string()).unwrap_or_else(|| "(preset/default)".to_string()),
            eff_clip_interm_domain,
            clip_interm_topk.map(|v| v.to_string()).unwrap_or_else(|| "(preset/default)".to_string()),
            clip_in_alpha_crown, clip_interm_prune, clip_interm_use_final_layer, eff_interm_transfer, eff_pgd_attack
        );
    }

    // Configure β-CROWN
    // Step 1: Start with defaults
    // Step 2: Apply preset if provided (benchmark-specific tuning)
    // Step 3: Override with explicit CLI flags only (Option::Some)
    let mut config = BetaCrownConfig::default();

    // Apply preset configuration first (establishes baseline from benchmark tuning)
    if let Some(ref preset) = preset_config {
        preset::apply_preset(&mut config, preset)?;
    }
    apply_vgg_abcrown_bound_mode(&mut config, vgg_abcrown_decision);
    apply_instance_overrides(&mut config, instance_overrides);
    apply_root_post_c_survivor_env(&mut config);

    // CLI flags override preset values ONLY when explicitly provided.
    // For Option<T> fields: only override when Some (CLI explicitly set).
    // For bool flags that default to false: only override when true (CLI explicitly
    // enables). When false (the default), preserve the preset value. This prevents
    // CLI defaults from silently overwriting preset-enabled features like relaxed_clip,
    // clip_interm_domain, etc. Bool flags that default to TRUE (pgd_attack,
    // enable_cuts) cannot use this rule — their `true` is indistinguishable from
    // the default — so they go through dedicated resolvers that let the preset win.
    if let Some(v) = max_domains {
        config.max_domains = v;
    } else if auto_input_split_selected
        && config.max_domains == BetaCrownConfig::default().max_domains
    {
        // Companion setting for auto-selected input splitting: input splitting fans
        // the input box into many subdomains and needs a larger frontier than the
        // ReLU-split default. Mirrors the `--max-domains 50000` the VNN-COMP runner
        // applied for input-split categories. Only applied when neither the CLI nor
        // a preset pinned max_domains. #ruarobot-clobber: the old assumption that
        // "a preset that set max_domains also set branching, so auto never fires"
        // is FALSE (safenlp_2024.yaml pins max_domains=4000 without a branching
        // method); raising the cap here clobbered the preset's load-bearing budget
        // split and starved the MIP escalation below its floor. The default-value
        // comparison detects a preset pin (same resolver pattern as the comment
        // above). SOUND: a domain cap only bounds search effort, never a verdict.
        config.max_domains = config
            .max_domains
            .max(branching::AUTO_INPUT_SPLIT_MAX_DOMAINS);
        info!(
            "Auto-branching: input splitting selected, max_domains set to {}",
            config.max_domains
        );
    }
    if let Some(v) = timeout {
        config.timeout = std::time::Duration::from_secs(v);
    }
    if let Some(v) = max_depth {
        config.max_depth = v;
    }
    // alpha: only override preset value when --no-alpha is explicitly passed.
    // Default (no_alpha=false) preserves preset's bound_prop_method setting.
    // This lets presets like acasxu_2023 disable alpha-CROWN via
    // `solver.bound_prop_method: crown` without requiring per-category CLI flags.
    if no_alpha {
        config.use_alpha_crown = false;
    }
    // crown_ibp/cuts: computed from graph-model logic, always apply
    config.use_crown_ibp = crown_ibp_effective;
    if let Some(ref heuristic) = cli_branching_heuristic {
        config.branching_heuristic = heuristic.clone();
    }
    if let Some(v) = fsb_candidates {
        config.fsb_candidates = v;
    }
    // Per-sub-domain α refinement in the input-split BaB loop. CLI overrides
    // preset overrides default (0 = off). SOUND: refined alphas are clamped to
    // [0,1]. Zero keeps frozen-alpha bound computation and leaves the reordered
    // production path unchanged; eager grouped screening may still finish an
    // already jointly verified domain earlier via its running lower floor.
    if let Some(v) = input_split_alpha_iterations {
        config.input_split_alpha_iteration = v;
    }
    if let Some(v) = input_split_lr_alpha {
        config.input_split_lr_alpha = v;
    }
    if let Some(v) = batch_size {
        config.batch_size = v;
    }
    config.parallel_children = parallel_children;
    // cuts: CLI `--enable-cuts` explicitly enables, `--no-cuts` explicitly
    // disables; otherwise the preset's `bab.cuts.enabled` (already applied by
    // apply_preset above) is preserved. Assigning the CLI value unconditionally
    // here silently clobbered preset-enabled cuts — the `ny vnncomp` entry point
    // always passes enable_cuts=false, so a preset's `bab.cuts.enabled: true`
    // never engaged. Forced off when the model class does not support cuts
    // (graph input splitting, #3813).
    config.enable_cuts =
        resolve_enable_cuts(config.enable_cuts, enable_cuts, no_cuts, cuts_supported);
    if let Some(v) = max_cuts {
        config.max_cuts = v;
    }
    if let Some(v) = min_cut_depth {
        config.min_cut_depth = v;
    }
    // Boolean flags: only override preset when CLI explicitly enables (true).
    // Default (false) preserves preset value.
    if enable_near_miss_cuts {
        config.enable_near_miss_cuts = true;
    }
    if let Some(v) = near_miss_margin {
        config.near_miss_margin = v;
    }
    if proactive_cuts {
        config.enable_proactive_cuts = true;
    }
    if let Some(v) = max_proactive_cuts {
        config.max_proactive_cuts = v;
    }
    if biccos_constraint_strengthening {
        config.enable_biccos_constraint_strengthening = true;
    }
    if let Some(v) = biccos_drop_ratio {
        config.biccos_drop_ratio = v;
    }
    if relaxed_clip {
        config.enable_relaxed_clip = true;
    }
    if let Some(v) = relaxed_clip_iterations {
        config.relaxed_clip_iterations = v;
    }
    if clip_interm_domain {
        config.enable_clip_interm_domain = true;
    }
    if let Some(v) = clip_interm_topk {
        config.clip_interm_topk = v;
    }
    if clip_in_alpha_crown {
        config.clip_in_alpha_crown = true;
    }
    if clip_interm_prune {
        config.clip_interm_prune = true;
    }
    if clip_interm_use_final_layer {
        config.clip_interm_use_final_layer = true;
    }
    if interm_transfer {
        config.enable_interm_transfer = true;
    }
    config.enable_la_warm_start = !no_la_warm_start;
    config.verify_upper_bound = verify_upper;
    // pgd_attack: `--pgd-attack` defaults to ON, so `pgd_attack == true` cannot
    // signal an explicit CLI choice. Assigning `true` unconditionally here
    // silently clobbered the `false` a preset's `attack.pgd_order: skip` had
    // just applied (yolo/nn4sys reserve that budget for BaB). The resolver
    // gives the preset precedence and keeps the default-on behavior when no
    // preset speaks; `--no-pgd-attack` still forces PGD off.
    config.enable_pgd_attack = resolve_pgd_attack(pgd_attack, preset_config.as_ref());
    if let Some(v) = pgd_restarts {
        config.pgd_restarts = v;
    }
    if let Some(v) = pgd_steps {
        config.pgd_steps = v;
    }
    if let Some(v) = lr_beta {
        config.beta_lr = v;
    }
    if let Some(v) = beta_iterations {
        config.beta_iterations = v;
    }
    if let Some(v) = beta_max_depth {
        config.beta_max_depth = v;
    }

    // α-CROWN config - only override when CLI explicitly provided
    if let Some(v) = alpha_iterations {
        config.alpha_config.iterations = v;
    }
    config.alpha_config.adaptive_skip = !no_adaptive_alpha_skip;
    if let Some(v) = alpha_skip_depth {
        config.alpha_config.adaptive_skip_depth_threshold = v;
    }
    config.alpha_config.fix_interm_bounds = !crown_ibp_intermediates;
    if let Some(v) = alpha_spsa_samples {
        config.alpha_config.spsa_samples = v;
    }
    if let Some(v) = alpha_lr {
        config.alpha_config.learning_rate = v;
    }
    if let Some(method) = alpha_gradient_method {
        config.alpha_config.gradient_method = method.into();
    }
    if let Some(opt) = alpha_optimizer {
        config.alpha_config.optimizer = opt.into();
    }

    maybe_set_alpha_output_constraints(&mut config.alpha_config, vnnlib_spec.as_ref())?;
    maybe_enable_invprop(
        &mut config.alpha_config,
        invprop,
        invprop_apply,
        invprop_share_gammas,
    )?;
    maybe_apply_build_batch_size_autotune_4354(
        &mut config,
        param_count,
        vnnlib_spec.as_ref(),
        is_graph_model,
        is_input_split_for_graph,
    );

    // Validate optimizer config after CLI flags and preset application.
    // Catches negative learning rates, NaN, and invalid Adam hyperparameters
    // before they silently corrupt bound optimization (#2942).
    config.validate()?;

    // Preset-scoped per-node CROWN-IBP time budget (#4413, #cgan-bn11-budget):
    // stamp the loaded graph so every downstream clone (dispatch, disjunctive
    // per-disjunct graphs, engine-configured copies) inherits the policy. The
    // all-None default (every preset that doesn't set the knobs) keeps the
    // built-in 2 s floor / 12 s cap constants byte-identically.
    if let BetaCrownModel::Graph(graph) = &mut model_net {
        graph.set_crown_ibp_per_node_time_budget(config.crown_ibp_per_node_time_budget());
    }

    // Auto-promote gpu_bab for graph models with supported heuristics on
    // single-objective specs. The DomainList frontier is non-regressing and
    // faster than the heap path for these lanes (#4406).
    // Disjunctive multi-clause specs are excluded — gpu_bab only checks 1
    // clause, causing status regression (#4409).
    let gpu_bab = should_auto_promote_gpu_bab(
        gpu_bab,
        is_graph_model,
        &config.branching_heuristic,
        vnnlib_spec.as_ref(),
    );

    let is_mip_only = complete_verifier == CompleteVerifierArg::Mip
        && matches!(&model_net, BetaCrownModel::Sequential(_));
    let run_upfront_pgd = effective_upfront_pgd_with_vgg(
        config.enable_pgd_attack,
        preset_config.as_ref(),
        vgg_abcrown_decision,
    );
    let mut dispatch_context = DispatchContext {
        model_path: &model,
        onnx_load_config: &onnx_load_config,
        model_net: &mut model_net,
        input: &input,
        config: &config,
        vnnlib_spec: vnnlib_spec.as_ref(),
        property: &property,
        epsilon,
        effective_threshold,
        verify_upper,
        output_dim,
        const_output_idx,
        has_relational,
        use_relu_split,
        gpu_bab,
        run_upfront_pgd,
        gemm_engine,
        compute_device: compute_device.clone(),
        allow_heuristic_logsoftmax,
        allow_heuristic_softmax,
        input_split_metrics_jsonl: input_split_metrics_jsonl.as_deref(),
        domain_batch_metrics_jsonl: domain_batch_metrics_jsonl.as_deref(),
        json,
        sigmoid_peeled,
        proof_opts: &proof_opts,
    };

    // CNF-recovery driver: a
    // SAT-encoded ReLU gadget (sat_relu) is decompiled to its source CNF and
    // decided EXACTLY by the `ay` CDCL solver — DRAT artifact on UNSAT, an
    // in-process-confirmed boolean witness on SAT (re-confirmed downstream by
    // the vnncomp ONNX-Runtime trusted-oracle gate). Placed BEFORE the MIP/BaB
    // fork: BaB is 0/100 on this family and the float-MIP fallback carries no
    // certificate. Fail-closed bit-exact detection: non-gadget models fall
    // through with zero behavior change. Disable with NY_NO_CNF_ROUTE=1.
    if let Some(vnnlib) = dispatch_context.vnnlib_spec {
        if let Some(result) = cnf_route::try_cnf_recovery(
            &*dispatch_context.model_net,
            dispatch_context.input.shape(),
            vnnlib,
            std::time::Instant::now() + dispatch_context.config.timeout,
        ) {
            output::output_result(
                &result,
                dispatch_context.property,
                dispatch_context.epsilon,
                dispatch_context.effective_threshold,
                dispatch_context.verify_upper,
                dispatch_context.json,
                dispatch_context.sigmoid_peeled,
            )?;
            return Ok(());
        }
    }

    if is_mip_only {
        return run_mip_only(&dispatch_context, mip_solver);
    }

    run_bab_with_fallback(&mut dispatch_context, complete_verifier, mip_solver)
}

/// Determine whether to auto-promote to the DomainList (`gpu_bab`) path.
///
/// Returns `true` when:
/// - `gpu_bab` was explicitly requested (CLI `--gpu-bab`), OR
/// - The model is a graph, the branching heuristic is supported
///   (`BoundImpact` or `InputSplit`), and the spec is not a multi-clause
///   disjunction (which the DomainList path does not yet iterate correctly,
///   see #4409).
///
/// Part of #4406: DomainList dispatch promotion for single-objective lanes.
/// Re: #2326 — promotion does not resolve queue/cache hardening.
pub(crate) fn should_auto_promote_gpu_bab(
    cli_gpu_bab: bool,
    is_graph_model: bool,
    heuristic: &BranchingHeuristic,
    vnnlib_spec: Option<&ny_onnx::vnnlib::VnnLibSpec>,
) -> bool {
    if cli_gpu_bab {
        return true;
    }
    if !is_graph_model {
        return false;
    }
    let supported_heuristic = matches!(
        heuristic,
        BranchingHeuristic::BoundImpact | BranchingHeuristic::InputSplit
    );
    if !supported_heuristic {
        return false;
    }
    // Guard: multi-clause disjunctive specs regress on gpu_bab (#4409).
    let is_multi_clause_disjunction = vnnlib_spec
        .is_some_and(|spec| spec.is_disjunction && spec.output_constraint_clauses.len() > 1);
    !is_multi_clause_disjunction
}

/// Parse a preset's `solver.mip.mip_solver` into the CLI arg type.
///
/// "scip" in a preset compiled WITHOUT the `mip-scip` feature degrades to
/// HiGHS with a warning rather than erroring: the preset is a per-category
/// performance tuning, and a scored VNN-COMP run must never die over a
/// missing optional backend (HiGHS solves the same problems, just slower).
/// Unknown values remain a hard preset error, mirroring
/// [`resolve_preset_complete_verifier`].
/// Whether the preset opts into the softmax "complex" decomposition
/// (`solver.alpha-crown.softmax: complex`, also accepted under
/// `bab.alpha-crown`), honoring the `NY_NO_SOFTMAX_COMPLEX=1` kill-switch
/// (disable-flag principle: the preset field is the opt-in, the env var the
/// emergency opt-out). Unknown values warn and keep the default direct-LSE
/// softmax relaxation.
fn resolve_preset_softmax_complex(preset_config: Option<&preset::PresetConfig>) -> bool {
    let requested = preset_config.and_then(|p| {
        p.solver
            .alpha_crown
            .softmax
            .as_deref()
            .or(p.bab.alpha_crown.softmax.as_deref())
    });
    match requested {
        None => false,
        Some(value) if value.eq_ignore_ascii_case("complex") => {
            if std::env::var_os("NY_NO_SOFTMAX_COMPLEX").is_some() {
                info!("NY_NO_SOFTMAX_COMPLEX set; softmax-complex rewrite disabled");
                false
            } else {
                true
            }
        }
        Some(other) => {
            warn!(
                "preset field solver.alpha-crown.softmax '{}' is not supported (expected 'complex') — using the default direct-LSE softmax relaxation",
                other
            );
            false
        }
    }
}

fn resolve_preset_mip_solver(
    preset_config: Option<&preset::PresetConfig>,
) -> Result<Option<MipSolverArg>> {
    match preset_config.and_then(|p| p.solver.mip.mip_solver.as_deref()) {
        None => Ok(None),
        Some("ay") => Ok(Some(MipSolverArg::AY)),
        // Legacy preset values from the pre-policy era (and alpha-beta-CROWN's
        // "gurobi"): all solving happens on ay (SOLVER POLICY). Warn so the
        // preset gets cleaned up, but do not fail a tuned category.
        Some(legacy @ ("highs" | "scip" | "gurobi")) => {
            warn!(
                "preset requests solver.mip.mip_solver: {legacy}, but ay is the only \
                 solver (SOLVER POLICY); solving on ay"
            );
            Ok(Some(MipSolverArg::AY))
        }
        Some(other) => Err(anyhow::anyhow!(
            "unsupported solver.mip.mip_solver '{other}': expected ay"
        )),
    }
}

/// Parse a preset's `general.complete_verifier` into the CLI arg type.
/// Unknown values are a hard preset error (silent fallback would quietly run the
/// wrong verifier for a category tuned to depend on it).
fn resolve_preset_complete_verifier(
    preset_config: Option<&preset::PresetConfig>,
) -> Result<Option<CompleteVerifierArg>> {
    match preset_config.and_then(|p| p.general.complete_verifier.as_deref()) {
        None => Ok(None),
        Some("auto") => Ok(Some(CompleteVerifierArg::Auto)),
        Some("bab") => Ok(Some(CompleteVerifierArg::Bab)),
        Some("mip") => Ok(Some(CompleteVerifierArg::Mip)),
        Some(other) => Err(anyhow::anyhow!(
            "unsupported general.complete_verifier '{other}': expected auto|bab|mip"
        )),
    }
}

/// Resolve the effective `enable_cuts`: explicit CLI enable/disable wins over the
/// preset value already applied to the config; model classes that do not support
/// cuts (graph input splitting, #3813) force them off regardless.
///
/// `preset_value` is `config.enable_cuts` after `apply_preset` (i.e. the preset's
/// `bab.cuts.enabled`, or the engine default `false`). `cli_enable` is the
/// `--enable-cuts` flag already collapsed with `--no-cuts` (`enable_cuts && !no_cuts`).
fn resolve_enable_cuts(
    preset_value: bool,
    cli_enable: bool,
    cli_disable: bool,
    cuts_supported: bool,
) -> bool {
    if !cuts_supported || cli_disable {
        return false;
    }
    cli_enable || preset_value
}

#[cfg(test)]
mod tests;
