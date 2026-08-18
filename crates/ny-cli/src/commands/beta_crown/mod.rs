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
//! - Cutting-plane configuration plumbing (verdict authority is quarantined)
//! - PGD attack for counterexample search
//! - GPU acceleration

#[cfg(feature = "mip")]
mod affine_invprop;
pub(crate) mod attack_arming;
#[cfg(feature = "mip")]
mod ay_tail_authority;
pub(crate) mod best_margin_export;
pub(crate) mod branching;
mod build_batch_size;
mod cell_enum;
mod cert_adapter;
#[cfg(feature = "mip")]
mod cgan_input_leaf;
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
#[cfg(feature = "mip")]
pub(crate) mod sign_space_falsify;
// Reachable from dispatch only through `star_candidate::run_dark_star_probe`, which is
// inert unless an operator arms `NY_STAR_DARK_SECONDS` and cannot emit a verdict.
#[cfg(feature = "mip")]
mod star_candidate;
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
        _model: Option<&std::path::Path>,
        _epsilon: f32,
        _threshold: f32,
        _overall_deadline: std::time::Instant,
        _warm_start_candidate: Option<&ndarray::ArrayD<f32>>,
        _mip_solver: crate::MipSolverArg,
        _reporting_start: std::time::Instant,
        _effective_treatment: &super::output::EffectiveTreatmentProjection,
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
pub(crate) use output::{
    begin_capture, end_capture, take_captured_json, take_captured_terminal_ingress,
    CapturedTerminalIngress,
};

/// Run the explicit-path Pensieve fractional-head diagnostic.
pub(crate) fn run_pensieve_fastpath_gap_research(
    onnx: &std::path::Path,
    vnnlib: &std::path::Path,
) -> Result<()> {
    frac_head::run_fastpath_gap_probe(onnx, vnnlib)
}

#[cfg(feature = "mip")]
fn require_graph_mip_fixture(bench_dir: &std::path::Path, relative: &str) -> Result<()> {
    let path = bench_dir.join(relative);
    anyhow::ensure!(
        path.is_file(),
        "missing graph-MIP fixture {}",
        path.display()
    );
    Ok(())
}

/// Run one explicitly selected graph-MIP corpus measurement.
///
/// The caller supplies the corpus root. Every file needed by the selected
/// probe is validated before solver work begins, so a missing checkout is a
/// command error rather than a skipped test.
#[cfg(feature = "mip")]
pub(crate) fn run_graph_mip_research(
    probe: &str,
    bench_dir: &std::path::Path,
    corpus_dir: Option<&std::path::Path>,
) -> Result<()> {
    anyhow::ensure!(
        bench_dir.is_dir(),
        "missing graph-MIP benchmark directory {}",
        bench_dir.display()
    );

    let require_instance_zero = || -> Result<()> {
        require_graph_mip_fixture(bench_dir, "onnx/original/ACASXU_run2a_2_4_batch_2000.onnx")?;
        require_graph_mip_fixture(
            bench_dir,
            "onnx/perturbed/ACASXU_run2a_2_4_batch_2000_perturbed_0.onnx",
        )?;
        require_graph_mip_fixture(bench_dir, "vnnlib/instance_0.vnnlib")
    };
    let require_selected_instance = || -> Result<()> {
        let f = std::env::var("NY_ISO_F_ONNX")
            .unwrap_or_else(|_| "onnx/original/ACASXU_run2a_3_8_batch_2000.onnx".into());
        let g = std::env::var("NY_ISO_G_ONNX").unwrap_or_else(|_| {
            "onnx/perturbed/ACASXU_run2a_3_8_batch_2000_perturbed_6.onnx".into()
        });
        let v =
            std::env::var("NY_ISO_VNNLIB").unwrap_or_else(|_| "vnnlib/instance_6.vnnlib".into());
        require_graph_mip_fixture(bench_dir, &f)?;
        require_graph_mip_fixture(bench_dir, &g)?;
        require_graph_mip_fixture(bench_dir, &v)
    };

    if probe != "emit-smtlib" {
        anyhow::ensure!(
            corpus_dir.is_none(),
            "--corpus-dir is only valid with the emit-smtlib probe"
        );
    }

    match probe {
        "diff-coupling" => {
            require_instance_zero()?;
            graph_mip_diff_coupling::research::measure_output_lp_shrink_on_real_instance0(
                bench_dir,
            );
        }
        "joint-relu-cuts" => {
            require_instance_zero()?;
            graph_mip_joint_relu_cuts::research::
                measure_joint_cut_output_lp_shrink_on_real_instance0(bench_dir);
        }
        "leaf-edge" => {
            require_instance_zero()?;
            graph_mip_leaf::research::live_decline_probe::probe_real_diffnet_edge_request(
                bench_dir,
            );
        }
        "obbt-box-width" => {
            require_graph_mip_fixture(bench_dir, "instances.csv")?;
            graph_mip_leaf::research::whole_net_diff_mip::measure_obbt_box_width_iso(bench_dir)?;
        }
        "whole-net-bb2b6088" => {
            require_selected_instance()?;
            graph_mip_leaf::research::whole_net_diff_mip::measure_iso_diff_whole_net_mip_bb2b6088(
                bench_dir,
            );
        }
        "k-vs-depth-ay" => {
            require_selected_instance()?;
            graph_mip_leaf::research::whole_net_diff_mip::measure_iso_k_vs_depth_and_ay_ceiling(
                bench_dir,
            );
        }
        "emit-smtlib" => {
            require_selected_instance()?;
            let out =
                corpus_dir.ok_or_else(|| anyhow::anyhow!("emit-smtlib requires --corpus-dir"))?;
            anyhow::ensure!(
                !out.as_os_str().is_empty(),
                "--corpus-dir must not be empty"
            );
            graph_mip_leaf::research::whole_net_diff_mip::emit_iso_diff_smtlib_corpus(
                bench_dir, out,
            );
        }
        "multineuron-root" => {
            require_selected_instance()?;
            anyhow::ensure!(
                std::env::var("NY_MULTINEURON").as_deref() == Ok("1")
                    && std::env::var("NY_MULTINEURON_MLP").as_deref() == Ok("1"),
                "multineuron-root requires NY_MULTINEURON=1 and NY_MULTINEURON_MLP=1"
            );
            graph_mip_leaf::research::whole_net_diff_mip::measure_iso_multineuron_root_tightening(
                bench_dir,
            );
        }
        "mscn-per-clause-128d" => {
            graph_mip::run_mscn_per_clause_table_research(bench_dir)?;
        }
        _ => anyhow::bail!(
            "unknown graph-MIP probe {probe:?}; expected diff-coupling, joint-relu-cuts, \
             leaf-edge, obbt-box-width, whole-net-bb2b6088, k-vs-depth-ay, emit-smtlib, \
             multineuron-root, or mscn-per-clause-128d"
        ),
    }
    Ok(())
}

use anyhow::Result;
use ny_core::GemmEngine;
use ny_propagate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownResult, BranchingHeuristic,
    VerificationArtifactAuthority, VggMaxPoolRewriteMode,
};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

use crate::preset;

use self::attack_arming::{shared_wgpu_attack_steering, AttackEngineSource, AttackSteering};
use self::dispatch::{run_bab_with_fallback, run_mip_only, DispatchContext};
use self::inputs::{check_needs_squeeze, create_input_bounds};
use self::invprop::{maybe_enable_invprop, maybe_set_alpha_output_constraints};
use self::model_load::{load_model, LoadedModel};
use self::routing::resolve_beta_crown_backend_request;
pub(crate) use self::routing::AUTO_BACKEND_GPU_MIN_INPUT_ELEMENTS;
use crate::commands::terminal_peel::AppliedTerminalPeel;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttackSteeringRoute {
    Disabled,
    SharedEngine,
    WgpuDevice,
    /// Reuse the exact qualified proof context; no second adapter/device or
    /// pipeline set may be constructed while this route is live.
    ProofDevice,
}

/// Attack-only view of the process-global GEMM accelerator.
///
/// The shared CUDA engine also implements several proof/resident capability
/// traits. Some attack lanes use the mere presence of those hooks to select a
/// batched point-VJP path, but CUDA does not yet implement the point-VJP
/// methods themselves. Advertising the raw engine would therefore enter an
/// unsupported path, repeatedly shrink the batch, and only then fall back.
///
/// Keep the two operations the attack actually benefits from and deliberately
/// leave every optional resident GPU/proof capability at
/// [`GemmEngine`]'s fail-closed defaults. This prevents the attack
/// handle from accidentally advertising the shared engine's resident
/// verdict-path capabilities; the dedicated attack-engine channel
/// (`attack_arming::AttackEngineSource`) remains the separate routing
/// boundary.
struct SharedAttackGemmOnly {
    inner: Arc<dyn GemmEngine>,
}

impl SharedAttackGemmOnly {
    fn new(inner: Arc<dyn GemmEngine>) -> Self {
        Self { inner }
    }
}

impl GemmEngine for SharedAttackGemmOnly {
    fn backend_provenance(&self) -> &'static str {
        self.inner.backend_provenance()
    }

    fn gemm_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        self.inner.gemm_f32(m, k, n, a, b)
    }

    fn gemm_f32_fast(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        self.inner.gemm_f32_fast(m, k, n, a, b)
    }
}

/// Pick the best falsification accelerator this host can actually offer.
///
/// Attack steering is verdict-neutral by construction (`attack_steering.rs`):
/// its floats only choose WHERE to look next, and every candidate still passes
/// the unchanged admission gates. So it must NOT be gated on the *proof*
/// backend — and gating it there was silently costing `sat` rows.
///
/// Historically the public quarantine changed a preset's `device: wgpu` into
/// `Cpu`; the old rule then read that fallback as "no accelerator" and disarmed
/// steering outright. The presets that request WGPU are exactly the
/// falsification-heavy ones (soundnessbench, cifar100, tinyimagenet, metaroom),
/// so the categories that most need accelerated attacks were the only ones
/// guaranteed not to get them. Measured on soundnessbench here: 25 of 28 rows
/// timed out with steering disarmed.
///
/// The public proof route now keeps the original request separate from the
/// effective backend. A qualified WGPU device is reused for attack VJPs; a
/// qualification refusal may use the separate verdict-neutral attack wrapper;
/// an explicit CPU request remains fully CPU.
///
/// #36 RESIDUAL (NVIDIA hosts): `SharedEngine` is NOT equivalent to
/// `WgpuDevice` for this consumer. [`SharedAttackGemmOnly`] deliberately hides
/// every optional resident capability (correctly: the CUDA engine advertises
/// hooks whose point-VJP methods it does not implement), so on an NVIDIA host
/// `gemm_engine.and_then(|e| e.as_gpu_crown_backward())` is `None` and BOTH
/// batched exact-VJP attack lanes — `graph_pgd_vjp_batched.rs:218` and
/// `graph_pgd_vjp_batched_disj.rs:103` — statically decline. Those are exactly
/// the lanes the wgpu presets are tuned around (metaroom's preset records the
/// batched exact-VJP disjunctive lane finishing a 30-restart budget in
/// 0.4s/3.6s and converting spec_idx_129/148 from timeout to sat). The
/// `AttackSteeringDevice` (WGPU) route DOES carry that hook, and this host has
/// a working adapter (NVIDIA GB10, IntegratedGpu, Vulkan).
///
/// The "redundant Vulkan graphics context costs more than it buys" rationale
/// predates #wallhugger-arming-cost (A6): arming now happens on a detached
/// background thread, so the context cost is off the critical path.
///
/// #36 RESIDUAL RESOLVED (2026-08-03): the residual is now MEASURED, and the
/// default moves to the capability-bearing route on NVIDIA hosts too.
/// soundnessbench model_0 at the official 150 s budget, same binary, same
/// preset, serial:
///
/// * `SharedEngine` → exact-VJP lane declines → sequential loop reaches
///   restart 0 / step 8 of the 121 s slice → **timeout at 145.4 s**;
/// * `WgpuDevice` → batched exact-VJP wave, K=64 at ~69 ms/step (1.1 ms per
///   restart-step) → **sat at 38.2 s** (wave restart 51, step 555).
///
/// #attack-steering-segv RESOLVED (2026-08-03): that route used to fault, and
/// this function used to answer `SharedEngine` unless a `NY_ATTACK_STEERING_WGPU=1`
/// opt-in. The fault was never in `WgpuDevice::new` — 12 of 12 gdb-captured
/// faults showed the MAIN thread inside `exit` → `_dl_fini` → `_dl_call_fini`
/// running libGLX_nvidia's destructors while the detached `ny-attack-arming`
/// thread was still building the device. Fast rows were the exposed class
/// because they publish a verdict inside the ~360 ms arming window. The scored
/// path now ends with `_exit` once the verdict is durable
/// (`exit_scored_instance_without_teardown`), which makes that race unreachable
/// — see that function for the captured stacks and the re-measured rate.
///
/// `NY_ATTACK_STEERING_WGPU=0` opts OUT (A/B and emergency lever); anything
/// else, including unset, takes the capability-bearing route.
/// The separate attack wrapper remains verdict-neutral and advertises no proof
/// authority. The `ProofDevice` arm is different: it reuses the already
/// qualified verdict context instead of constructing any wrapper.
fn attack_steering_route(
    requested_backend: BackendArg,
    qualified_wgpu_proof_active: bool,
    wgpu_probe_skipped: bool,
    accelerator_requested: bool,
) -> AttackSteeringRoute {
    if qualified_wgpu_proof_active {
        return AttackSteeringRoute::ProofDevice;
    }
    if requested_backend == BackendArg::Cpu && !accelerator_requested {
        // Genuine CPU request: honour it for attacks too.
        return AttackSteeringRoute::Disabled;
    }
    if wgpu_probe_skipped && attack_steering_wgpu_opt_out() {
        // Explicit opt-out on a probe-skipped host: reuse the process-global
        // CUDA engine instead of opening a second graphics context. Costs the
        // exact-VJP lanes (soundnessbench falls back to 1/50), so this exists
        // for A/B measurement, not for scoring.
        AttackSteeringRoute::SharedEngine
    } else {
        AttackSteeringRoute::WgpuDevice
    }
}

/// Opt-OUT lever for the #36 residual (see [`attack_steering_route`]).
/// `NY_ATTACK_STEERING_WGPU=0` keeps the shared engine on a host whose adapter
/// probe was skipped; anything else (including unset) takes the WGPU route.
///
/// Exact-literal `0` only: a malformed or non-Unicode value must not silently
/// disarm the accelerator on a scored run.
pub(crate) fn attack_steering_wgpu_opt_out() -> bool {
    std::env::var("NY_ATTACK_STEERING_WGPU").ok().as_deref() == Some("0")
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

/// What a validated preset's executable initial schedule says about engine PGD
/// enablement. Unimplemented/uncontracted orders resolve to `None` here and
/// are rejected by semantic preset validation before execution.
fn preset_pgd_enabled(preset_config: Option<&preset::PresetConfig>) -> Option<bool> {
    preset::resolve_initial_pgd_schedule(preset_config?)
        .ok()
        .flatten()
        .map(|schedule| !matches!(schedule, preset::ResolvedInitialPgdSchedule::Disabled))
}

fn preset_uses_input_bab_pgd_schedule(preset_config: Option<&preset::PresetConfig>) -> bool {
    preset_config.is_some_and(|preset| {
        matches!(
            preset::resolve_initial_pgd_schedule(preset),
            Ok(Some(preset::ResolvedInitialPgdSchedule::InputBab))
        )
    })
}

fn preset_uses_deferred_pgd_schedule(preset_config: Option<&preset::PresetConfig>) -> bool {
    preset_config.is_some_and(|preset| {
        matches!(
            preset::resolve_initial_pgd_schedule(preset),
            Ok(Some(preset::ResolvedInitialPgdSchedule::Deferred))
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeferredPgdOwner {
    None,
    InternalEngine,
    OuterWrapper,
}

fn resolve_deferred_pgd_owner(
    deferred_schedule: bool,
    direct_internal_consumer_available: bool,
    outer_consumer_available: bool,
) -> DeferredPgdOwner {
    if !deferred_schedule {
        DeferredPgdOwner::None
    } else if outer_consumer_available {
        DeferredPgdOwner::OuterWrapper
    } else if direct_internal_consumer_available {
        DeferredPgdOwner::InternalEngine
    } else {
        DeferredPgdOwner::None
    }
}

fn require_deferred_pgd_consumer(
    preset_config: Option<&preset::PresetConfig>,
    pgd_enabled: bool,
    model_is_graph: bool,
    late_sequential_conjunction_graph_upgrade: bool,
    mip_only_route: bool,
    direct_internal_consumer_available: bool,
    outer_consumer_available: bool,
) -> Result<()> {
    let verifier_route = if mip_only_route {
        "MIP-only"
    } else if model_is_graph {
        "Graph"
    } else if late_sequential_conjunction_graph_upgrade {
        "late Sequential-to-Graph"
    } else {
        "Sequential"
    };
    let deferred_schedule = pgd_enabled && preset_uses_deferred_pgd_schedule(preset_config);
    if deferred_schedule
        && resolve_deferred_pgd_owner(
            deferred_schedule,
            direct_internal_consumer_available,
            outer_consumer_available,
        ) == DeferredPgdOwner::None
    {
        anyhow::bail!(
            "attack.pgd_order=after requires a reachable post-BaB attack consumer; the selected \
             {verifier_route} route has neither an internal nor enabled outer phase (use \
             attack.ny_pgd_order_compat=upfront or enable the VNN-COMP post-BaB attack)"
        );
    }
    Ok(())
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

/// Remove the engine's post-BaB PGD reservation when engine PGD is disabled.
///
/// `attack.pgd_order: skip` and `--no-pgd-attack` must disable both execution
/// and its scheduling-only slice. This does not alter
/// `vnncomp_post_bab_attack`: the outer wrapper is resolved independently and
/// may still spend genuine leftover wall time, but it cannot silently acquire
/// an engine reservation from a disabled schedule.
fn effective_engine_phase_budget(
    enable_pgd_attack: bool,
    mut phase_budget: ny_propagate::PhaseBudgetConfig,
) -> ny_propagate::PhaseBudgetConfig {
    if !enable_pgd_attack {
        phase_budget.post_bab_pgd_fraction = 0.0;
    }
    phase_budget
}

fn effective_upfront_pgd(
    enable_pgd_attack: bool,
    preset_config: Option<&preset::PresetConfig>,
) -> bool {
    enable_pgd_attack
        && !preset_uses_input_bab_pgd_schedule(preset_config)
        && !preset_uses_deferred_pgd_schedule(preset_config)
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
        && !preset_uses_deferred_pgd_schedule(preset_config)
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
/// SOUNDNESS: requesting certificate emission can conservatively withhold a
/// `Verified` verdict from an optimization whose proof is not representable by
/// the current external certificate format. It can never create or strengthen
/// a verdict. The float-soundness posture (f64 intermediates, directed rounding,
/// conservative verdict translation) is baseline and is NEVER gated by these
/// options.
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
    /// (`<model-stem>.cert.json` alongside the model).
    pub certificate_path: Option<PathBuf>,
    /// Removed CLI compatibility flag. A `true` value is rejected before model
    /// loading; unsound GPU bounds cannot decide a verdict.
    pub allow_unsound_gpu_crown: bool,
}

impl ProofOpts {
    fn validate(&self) -> Result<()> {
        if self.allow_unsound_gpu_crown {
            anyhow::bail!(
                "--allow-unsound-gpu-crown is disabled: the public WGPU proof integration \
                 is quarantined and unsound GPU bounds cannot be admitted to a verification verdict"
            );
        }
        Ok(())
    }

    /// Whether a proof-carrying certificate should be emitted for a `Verified`
    /// verdict, resolving the `emit_certificate` override against
    /// `competition_mode`.
    pub(crate) fn should_emit_certificate(&self) -> bool {
        self.emit_certificate.unwrap_or(!self.competition_mode)
    }

    /// Resolve the frontend artifact policy into the verifier's typed authority.
    ///
    /// The mapping is exact: every request that can reach certificate emission
    /// carries `CertificateExport`; competition/explicit no-certificate runs
    /// carry `VerdictOnly`.
    pub(crate) fn verification_artifact_authority(&self) -> VerificationArtifactAuthority {
        if self.should_emit_certificate() {
            VerificationArtifactAuthority::CertificateExport
        } else {
            VerificationArtifactAuthority::VerdictOnly
        }
    }

    /// Whether the process-global soundness gate is engaged.
    ///
    /// This is unconditional: no CLI or competition-mode policy may admit the
    /// unsound fast GPU f32 path to a verdict.
    pub(crate) fn sound_gpu_crown_required(&self) -> bool {
        true
    }
}

/// One immutable read of a VNN-COMP preset.
///
/// Loaded snapshots have passed every preset-only semantic validator. The outer
/// wrapper resolves scheduling from this value and passes the exact same loaded
/// snapshot into the β-CROWN handler. An invalid snapshot is retained so the
/// VNN-COMP runner can return a sound `unknown` before every wrapper verdict
/// lane. Interactive CLI calls do not provide a snapshot and retain the
/// historical load-by-path behavior.
#[derive(Debug, Clone)]
pub(crate) enum BetaCrownPresetSnapshot {
    Loaded(Box<preset::PresetConfig>),
    Invalid(String),
}

/// Validate every preset-only semantic decision that can fail before model
/// execution.
///
/// VNN-COMP calls this while freezing the preset so no outer SAT/UNSAT fast
/// path can publish a verdict from a YAML-valid preset that the in-process
/// verifier would later reject. This intentionally includes both verifier
/// config application and ONNX-loader policy, plus the two router-owned enum
/// fields resolved outside `preset::apply_preset`.
fn validate_beta_crown_preset(preset_config: &preset::PresetConfig) -> Result<()> {
    preset::validate_preset(preset_config)?;
    preset::build_onnx_load_config(preset_config)?;
    resolve_preset_complete_verifier(Some(preset_config))?;
    resolve_preset_mip_solver(Some(preset_config))?;
    Ok(())
}

impl BetaCrownPresetSnapshot {
    pub(crate) fn load(path: &std::path::Path) -> Self {
        match preset::load_preset(path) {
            Ok(config) => match validate_beta_crown_preset(&config) {
                Ok(()) => Self::Loaded(Box::new(config)),
                Err(error) => Self::Invalid(format!(
                    "Invalid preset semantics in {}: {error:#}",
                    path.display()
                )),
            },
            Err(error) => Self::Invalid(error.to_string()),
        }
    }

    pub(crate) fn loaded(&self) -> Option<&preset::PresetConfig> {
        match self {
            Self::Loaded(config) => Some(config.as_ref()),
            Self::Invalid(_) => None,
        }
    }

    pub(crate) fn invalid_error(&self) -> Option<&str> {
        match self {
            Self::Loaded(_) => None,
            Self::Invalid(error) => Some(error),
        }
    }
}

/// Per-instance, typed overrides owned by the in-process VNN-COMP router.
/// Interactive CLI calls always pass `Default::default()`. Keeping this typed
/// avoids process-global environment mutation after an exact instance policy
/// has already been resolved.
#[derive(Debug, Clone, Default)]
pub(crate) struct BetaCrownInstanceOverrides {
    /// Arm bounded sparse root CROWN for the sealed adaptive-release route.
    pub(crate) root_sparse_interm_crown: bool,
    /// Absolute outer deadline already anchored by the in-process VNN-COMP
    /// router. `None` preserves the interactive/historical timeout ledger.
    pub(crate) authoritative_deadline: Option<std::time::Instant>,
    /// Absolute end of the internal verifier phase when VNN-COMP owns a
    /// compat-free deferred PGD tail. The wrapper freezes this from the
    /// instance-level clock after charging upfront work and fixed tail
    /// reserves. The handler may cap it further, but must never reconstruct it
    /// from a later setup-completion time.
    pub(crate) outer_deferred_internal_deadline: Option<std::time::Instant>,
    /// Whether the in-process VNN-COMP wrapper has an executable post-BaB
    /// attack consumer. `None` preserves the historical interactive budget
    /// ledger, which retained the small-budget tail.
    pub(crate) post_bab_wrapper_attack_enabled: Option<bool>,
    /// VNN-COMP's shared post-BaB/β-CROWN preset read. `None` keeps the
    /// interactive handler's historical path-based load.
    pub(crate) preset_snapshot: Option<BetaCrownPresetSnapshot>,
    /// The in-process VNN-COMP router observed the exact direct-MIP-first gate
    /// on a positive, short `safenlp_2024` scored budget. This is intent only:
    /// dispatch still requires the exact model/property shape, AUTO policy, a
    /// live bounded deadline, and the independent shared-prefix gate.
    pub(crate) safenlp_direct_mip_first: bool,
    /// The VNN-COMP router admitted the exact default-dark traffic-category
    /// terminal-Softmax peel treatment. The loader still applies its existing
    /// fail-closed all-constraints rewrite and records whether it actually
    /// peeled a layer.
    pub(crate) traffic_terminal_softmax_peel: bool,
    /// Exact VNN-COMP category route for the default-dark imgSz32 cGAN input
    /// leaf proof.  The typed variant can only be minted by the in-process
    /// category router; the oracle independently authenticates the model,
    /// property, normalized graph, and every requested objective row.
    pub(crate) cgan_input_leaf_route: Option<CganInputLeafRoute>,
}

/// Category authority carried into the imgSz32 cGAN input-leaf attachment.
///
/// This intentionally has no string-bearing or generic variant: a future cGAN
/// category must earn a separate route rather than inheriting the 2023 seal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CganInputLeafRoute {
    Cgan2023,
}

pub(crate) use super::cgan_status::CGAN_DEPTH_TWO_PRODUCTION_MODE;

fn resolve_preset_config(
    preset_path: Option<&std::path::Path>,
    preset_snapshot: Option<&BetaCrownPresetSnapshot>,
) -> Result<Option<preset::PresetConfig>> {
    let config = match preset_snapshot {
        Some(BetaCrownPresetSnapshot::Loaded(config)) => Some(config.as_ref().clone()),
        Some(BetaCrownPresetSnapshot::Invalid(error)) => return Err(anyhow::anyhow!("{error}")),
        None => preset_path.map(preset::load_preset).transpose()?,
    };
    if let Some(config) = config.as_ref() {
        validate_beta_crown_preset(config)?;
    }
    Ok(config)
}

/// Combine the CLI-local verifier timeout with an optional deadline that was
/// anchored before in-process VNN-COMP setup.
///
/// The earlier deadline is authoritative, but it must never lengthen a
/// deliberately shorter inner timeout (for example, a reserved post-BaB
/// attack slice). `None` plus a zero timeout preserves the interactive
/// unbounded contract.
fn resolve_overall_deadline(
    verification_start: std::time::Instant,
    configured_timeout: std::time::Duration,
    authoritative_deadline: Option<std::time::Instant>,
) -> Result<Option<std::time::Instant>> {
    let configured_deadline = if configured_timeout.is_zero() {
        None
    } else {
        Some(
            verification_start
                .checked_add(configured_timeout)
                .ok_or_else(|| anyhow::anyhow!("timeout is too large for the platform clock"))?,
        )
    };
    Ok(match (configured_deadline, authoritative_deadline) {
        (Some(configured), Some(authoritative)) => Some(configured.min(authoritative)),
        (Some(configured), None) => Some(configured),
        (None, Some(authoritative)) => Some(authoritative),
        (None, None) => None,
    })
}

fn resolve_internal_authority_deadline(
    overall_deadline: Option<std::time::Instant>,
    outer_wrapper_owns_deferred_pgd: bool,
    frozen_outer_deferred_deadline: Option<std::time::Instant>,
) -> Result<Option<std::time::Instant>> {
    if !outer_wrapper_owns_deferred_pgd {
        return Ok(overall_deadline);
    }
    let frozen = frozen_outer_deferred_deadline.ok_or_else(|| {
        anyhow::anyhow!(
            "VNN-COMP outer deferred PGD ownership requires a frozen absolute internal deadline"
        )
    })?;
    Ok(Some(
        overall_deadline.map_or(frozen, |deadline| deadline.min(frozen)),
    ))
}

fn apply_instance_overrides(
    config: &mut BetaCrownConfig,
    instance_overrides: &BetaCrownInstanceOverrides,
) {
    if instance_overrides.root_sparse_interm_crown {
        // Exact per-instance performance route. The root policy's raw
        // `NY_ROOT_SPARSE_INTERM_CROWN=0` resolver remains the final kill
        // switch and can still disable this typed-on config.
        config.root_sparse_interm_crown = true;
    }
    if instance_overrides.cgan_input_leaf_route == Some(CganInputLeafRoute::Cgan2023) {
        // The engine seam is independently default-dark. Only the exact typed
        // VNN-COMP route may ask it to consult; construction still declines on
        // every model/property/profile/provenance mismatch.
        config.input_split_input_leaf_oracle = true;
    }
}

/// Frontend policy for terminal-activation peeling.
///
/// The historical interactive flag retains its broader legacy behavior.  The
/// typed traffic route is deliberately distinct and stricter: it may peel only
/// an authenticated, single-group terminal Softmax.  Keeping the source typed
/// prevents a competition-only request from silently inheriting the legacy
/// LogSoftmax/Sigmoid surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TerminalPeelPolicy {
    Off,
    InteractiveLegacy,
    TrafficSoftmaxSingleGroup,
}

impl TerminalPeelPolicy {
    fn requested(self) -> bool {
        self != Self::Off
    }
}

fn terminal_peel_policy(
    cli_requested: bool,
    instance_overrides: &BetaCrownInstanceOverrides,
) -> TerminalPeelPolicy {
    // A typed competition route always keeps its narrow soundness fence, even
    // if a future caller accidentally also forwards the legacy CLI bit.
    if instance_overrides.traffic_terminal_softmax_peel {
        TerminalPeelPolicy::TrafficSoftmaxSingleGroup
    } else if cli_requested {
        TerminalPeelPolicy::InteractiveLegacy
    } else {
        TerminalPeelPolicy::Off
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
/// - Cutting-plane configuration plumbing (verdict authority is quarantined)
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
    max_queue_bytes: Option<usize>,
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
    deferred_pgd_consumer_available: bool,
    instance_overrides: BetaCrownInstanceOverrides,
) -> Result<()> {
    // Keep execution attribution scoped to this command, including captured
    // in-process VNN-COMP calls. Starting before validation guarantees that an
    // early return clears any prior solve's observations.
    let _execution_telemetry_run = ny_propagate::execution_telemetry::begin_run();
    proof_opts.validate()?;
    let terminal_peel_policy = terminal_peel_policy(peel_last_softmax_layer, &instance_overrides);
    let peel_last_softmax_layer = terminal_peel_policy.requested();

    #[cfg(feature = "mip")]
    graph_mip::install_imb_ay_tail_certificate_oracle();

    // Resolve effective backend from CLI flags and preset.
    // Precedence: --backend > --gpu > preset general.device > CPU default.
    let enable_cuts = enable_cuts && !no_cuts;

    // Load the preset by path for interactive calls. The VNN-COMP wrapper
    // supplies its already-resolved loaded-or-invalid snapshot so post-BaB
    // scheduling and β-CROWN verification cannot observe different versions
    // of one file.
    let preset_config = resolve_preset_config(
        preset.as_deref(),
        instance_overrides.preset_snapshot.as_ref(),
    )?;
    if let (Some(preset_path), Some(loaded)) = (preset.as_deref(), preset_config.as_ref()) {
        info!("Loaded preset config: {}", preset_path.display());
        // #preset-contract (design S3). `load_preset` proves every key is a key
        // the schema knows; this proves the ENGINE will act on it. A key that
        // parses, lands in `BetaCrownConfig`, and is then read by nothing is
        // indistinguishable from a working one everywhere except the scoreboard.
        // Reports at load, before any budget is spent, and only refuses to start
        // for a verdict-affecting field with no argued failure direction
        // (`preset::contract`). `NY_PRESET_STRICT=0` silences it entirely.
        preset::enforce_preset_contract(preset_path, loaded)?;
    }
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
    // general.device still take precedence. Keep that SOURCE alongside the
    // selected backend: runtime qualification may subsequently fall back, and
    // the evidence must not confuse a preset/auto request with execution.
    // The capability hint defaults to compile-time wgpu availability; the scored
    // VNN-COMP path narrows it with the `GPU_AVAILABLE` runtime probe so we never
    // even *prefer* GPU on a box that has none.
    let gpu_available = gpu_available.unwrap_or_else(ny_gpu::wgpu_backend_compiled);
    let backend_request = resolve_beta_crown_backend_request(
        backend,
        gpu,
        preset_device,
        peeked_input_count,
        gpu_available,
    );
    let auto_wgpu_candidate = backend_request.backend == BackendArg::Wgpu;
    // Proof authority policy may rewrite an AUTO WGPU candidate to CPU (for
    // example NY_WGPU_CROWN=0 or NY_CUDA_CROWN=1). Preserve the pre-policy
    // performance intent for verdict-neutral attack steering: a proof-only
    // kill switch must not silently disarm falsification acceleration.
    let attack_backend_request = backend_request;
    let backend_request = crate::commands::backend::resolve_automatic_wgpu_request(
        backend_request,
        auto_wgpu_candidate,
    )?;
    let requested_backend = backend_request.backend;
    if let Some(reason) = backend_request.selection_reason {
        info!(
            "Auto-backend: selected {} ({}); input_elements={:?}",
            requested_backend, reason, peeked_input_count
        );
        if !json {
            println!(
                "Auto-backend: selected {} ({}); input_elements={}",
                requested_backend,
                reason,
                peeked_input_count.map_or_else(|| "unknown".to_string(), |c| c.to_string())
            );
        }
    } else if backend.is_none() && !gpu && requested_backend != BackendArg::Cpu {
        info!("Preset selected {} backend", requested_backend);
    }

    // Engage the proof-consumer soundness filter before constructing an
    // accelerator. The typed constructor then creates exactly one WGPU context,
    // runs all five live qualification rungs on it, and returns that same context
    // only on success. Every refusal falls back to a concrete CPU proof device
    // and emits NY-HARNESS: BACKEND-OVERRIDE unconditionally inside the shared
    // resolver (including JSON/scored invocations).
    ny_propagate::set_sound_gpu_crown_required(proof_opts.sound_gpu_crown_required());
    let proof_resolution =
        crate::commands::backend::resolve_proof_backend(backend_request, "beta-crown")?;
    let proof_backend_receipt = proof_resolution.receipt;
    let qualified_wgpu_proof_active = proof_backend_receipt.qualified_wgpu_active();
    let effective_backend = proof_backend_receipt.effective;
    let mut compute_device = Some(Arc::new(proof_resolution.device));
    // #charged-metal-engagement: publish the EXACT qualified proof device into
    // the process-global sound CROWN engine slots (a clone of this Arc — no
    // second WGPU context) so borrow-only consumers — the deadline-
    // preinitialized sequential route, the margin-row batch seam, the resident
    // cut shadow, the wide lanes — can see it. Runs at qualification time,
    // BEFORE verifier construction creates finite deadline authority. No-op
    // unless the receipt proves live qualification; `main`'s earlier CUDA
    // installs keep first-install precedence.
    if let Some(device) = &compute_device {
        let engine: Arc<dyn GemmEngine> = device.clone();
        crate::commands::backend::register_qualified_wgpu_proof_engine(
            &proof_backend_receipt,
            &engine,
        );
    }

    // These A/B levers only suppress the optional CPU engine-presence handle.
    // They may never discard a successfully qualified WGPU proof device.
    if !qualified_wgpu_proof_active
        && (std::env::var("NY_ROOT_JOINT_INTERM_ALPHA").ok().as_deref() == Some("1")
            || std::env::var("NY_CPU_COMPUTE_ENGINE").ok().as_deref() == Some("0"))
    {
        compute_device = None;
    }
    info!(
        requested_backend = %proof_backend_receipt.requested,
        request_source = proof_backend_receipt.request_source.as_str(),
        effective_backend = %proof_backend_receipt.effective,
        qualification = proof_backend_receipt.qualification.as_str(),
        proof_backend_provenance = %proof_backend_receipt.provenance,
        "resolved beta-crown proof backend"
    );

    // #attack-steering-unquarantine + #wallhugger-arming-cost: the quarantine
    // above governs PROOF execution; falsification steering keeps its
    // accelerator (see `attack_arming` for the full contract). Arming starts
    // HERE — before model loading — on a background thread, so the
    // `WgpuDevice::new` construction cost (adapter + device + ~20 pipeline
    // compiles) overlaps model load / bound setup instead of being paid
    // serially at instance start, where it tipped ≤5s-margin banked unsats
    // into timeout (the recovery sweep's 5 wall-huggers, all ~97s). Attack
    // lanes take the engine ONLY if arming has finished by their take-point
    // and proceed un-steered otherwise; construction failure falls back to
    // CPU steering and must never fail the run. Verdict-neutral by
    // construction: the handle is threaded ONLY into attack call sites
    // (`AttackEngineSource`), never into bound/precheck/BaB work, and every
    // candidate still passes the unchanged admission gates (engine confirm +
    // global-box guard; trusted-ORT + true-f64 on the scored path).
    let steering_route = attack_steering_route(
        attack_backend_request.backend,
        qualified_wgpu_proof_active,
        crate::compute_backend::detect().wgpu_probe_skipped,
        attack_backend_request.backend == BackendArg::Wgpu,
    );
    // #alpha-steering-proposal: on WGPU-adapter hosts, install the α-gradient
    // PROPOSAL factory for the DAG α-CROWN margin-gradient lane. Lazy: the
    // capability wrapper is constructed only on the first armed
    // margin-gradient iteration that finds no verdict-authority resident
    // backend — unarmed runs pay nothing. Default-yes on any adapter because
    // the consumer is proposal-grade by construction (gradients only steer
    // α∈[0,1]; the certified CPU fold evaluates every iterate — design I3).
    // NOT installed on the SharedEngine (CUDA) route: there the resident
    // joint adjoint already passes the authority filter and the proposal
    // seam is never consulted, keeping GB10 routing byte-identical.
    //
    // #attack-steering-conjunctive keeps that byte-identity DELIBERATE rather
    // than incidental. Moving the ATTACK route to the WGPU device on NVIDIA
    // hosts must not drag these two unrelated seams along with it: they are
    // properties of the host's PROOF/α regime, not of falsification steering,
    // and neither has been measured on a CUDA host. Gate them on the original
    // condition — a host whose adapter probe actually ran.
    //
    // The α, FL-value, and attack wrappers below are distinct CAPABILITY views,
    // not distinct graphics contexts. Their `new_wgpu` constructors all borrow
    // ny-gpu's one process-shared ordinary `Arc<WgpuDevice>`, so a CPU/fallback
    // proof route cannot retain three independent pipeline/device allocations.
    let install_wgpu_proposal_seams = !qualified_wgpu_proof_active
        && matches!(steering_route, AttackSteeringRoute::WgpuDevice)
        && !crate::compute_backend::detect().wgpu_probe_skipped;
    if install_wgpu_proposal_seams {
        ny_propagate::alpha_gradient_steering::set_alpha_gradient_steering_factory(|| {
            match ny_gpu::GradientSteeringDevice::new_wgpu() {
                Ok(device) => Some(Arc::new(device) as Arc<dyn GemmEngine>),
                Err(error) => {
                    warn!(
                        %error,
                        "α-gradient steering engine unavailable; margin-gradient \
                         lane keeps its bounded local fallback (#alpha-steering-proposal)"
                    );
                    None
                }
            }
        });
        // #fl-value-gpu-tier: on the same wgpu-adapter hosts, install the
        // deadline-bounded f32 value-GEMM factory for the forward-linear
        // seam. Lazy + deadline-safe (background construction, bounded
        // admission wait); consulted ONLY when the f32 seam is armed
        // (`NY_FORWARD_LINEAR_F32`, unchanged) AND a finite deadline is
        // present, and the call site charges `γ^f32·S` + FTZ for its values.
        // No env flag gates the registration itself (I10): the engine's
        // measured size threshold and the seam gate decide per call. The rate
        // probe measures THROUGH this chain, so FL admission re-prices
        // automatically wherever this tier is faster.
        // RESTORED 2026-08-02 now that `ca4740cb` re-landed the producer. The
        // refusing stub my build hotfix put here had to go: installation is
        // first-install-wins (`FACTORY.set` on a `OnceLock`), and `main.rs`
        // only registers behind `if !compute_backend::detect().wgpu_probe_skipped`
        // — so on any host that skips the probe the stub would have won the
        // race and silently suppressed the re-landed tier.
        ny_propagate::fl_value_gemm::set_fl_value_gemm_factory(|| {
            match ny_gpu::FlValueGemmDevice::new_wgpu() {
                Ok(device) => Some(Arc::new(device) as Arc<dyn GemmEngine>),
                Err(error) => {
                    warn!(
                        %error,
                        "FL-value wgpu f32 engine unavailable; forward-linear \
                         value GEMMs keep the tiled CPU tiers (#fl-value-gpu-tier)"
                    );
                    None
                }
            }
        });
    }
    // The capability-bearing WGPU route shares one process-global asynchronous
    // armer with wrapper attack pre-waves. Its narrow wrapper and the α/FL
    // wrappers above all resolve to ny-gpu's same ordinary WGPU context.
    // Disabled/shared-engine routes remain call-local: they carry no graphics
    // context and must not initialize the global WGPU slot. Reuse prevents a
    // timed-out wrapper init from being discarded and avoids constructing a
    // second WGPU context for verifier PGD.
    let local_attack_steering = match steering_route {
        AttackSteeringRoute::Disabled => Some(AttackSteering::disarmed()),
        AttackSteeringRoute::SharedEngine => {
            // On NVIDIA, reuse the process-global CUDA/CPU engine installed by
            // main. Constructing WGPU here would reopen the Vulkan graphics
            // context allocation that backend detection deliberately avoided.
            // The engine already exists, so this route is ready immediately.
            match ny_propagate::fast_f32_gemm::shared_engine() {
                Some(engine) => {
                    info!(
                        backend = engine.backend_provenance(),
                        "attack-steering shared GEMM engine armed; optional resident/VJP \
                         capabilities masked and redundant WGPU device avoided"
                    );
                    Some(AttackSteering::ready(Arc::new(SharedAttackGemmOnly::new(
                        engine,
                    ))))
                }
                None => {
                    warn!(
                        "attack-steering shared engine unavailable; redundant WGPU device \
                         remains blocked and attack lanes fall back to CPU steering"
                    );
                    Some(AttackSteering::disarmed())
                }
            }
        }
        AttackSteeringRoute::WgpuDevice => None,
        AttackSteeringRoute::ProofDevice => {
            let proof_device = compute_device
                .as_ref()
                .expect("qualified WGPU route retains its proof device");
            info!(
                backend = proof_device.backend_provenance(),
                "attack steering reuses the qualified WGPU proof context"
            );
            Some(AttackSteering::ready(
                Arc::clone(proof_device) as Arc<dyn GemmEngine>
            ))
        }
    };
    let attack_steering = local_attack_steering
        .as_ref()
        .unwrap_or_else(|| shared_wgpu_attack_steering());
    let attack_engine_source = AttackEngineSource::Arming(attack_steering);

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
        applied_terminal_peel,
        auto_branching,
    } = {
        let (loaded_model, routed_relu_split) = load_model(
            &model,
            &onnx_load_config,
            property.as_deref(),
            terminal_peel_policy,
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
    // Legacy engine seams need only the Sigmoid distinction; the full typed
    // receipt remains attached to effective treatment for witness publication.
    let sigmoid_peeled = applied_terminal_peel.is_sigmoid();
    // #fl-alpha-composition: `model.forward_alpha_surrogate` is the
    // graph-generic sibling of the authored-graph spec-alpha key — same typed
    // lever
    // (#w4-root-alpha-opt certified rebuild), authority over the LOADED graph,
    // so no `preserve_raw` admission (see the preset field docs).
    let forward_linear_spec_alpha = preset_config.as_ref().is_some_and(|preset| {
        preset.model.forward_linear_spec_alpha == Some(true)
            || preset.model.forward_alpha_surrogate == Some(true)
    });
    match &mut model_net {
        BetaCrownModel::Graph(graph) => {
            graph.set_forward_linear_spec_alpha(forward_linear_spec_alpha);
        }
        BetaCrownModel::Sequential(_) if forward_linear_spec_alpha => {
            anyhow::bail!(
                "model.forward_linear_spec_alpha (legacy alias: \
                 model.cgan_forward_alpha_surrogate) / model.forward_alpha_surrogate require \
                 GraphNetwork routing; refusing an enabled but unreachable proof lane"
            );
        }
        BetaCrownModel::Sequential(_) => {}
    }

    // Consume the model-class-aware auto-branching decision (if `auto` was
    // requested and no preset owned branching). `load_model` already used it for
    // its own conv/DAG routing; here we mirror it into the config-facing state.
    if let Some(resolved) = auto_branching.as_ref() {
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
        cli_branching_heuristic = Some(resolved.heuristic.clone());
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

    // Create the proof-engine reference from the exact retained device. A
    // qualified WGPU wrapper exposes only sound CROWN backward; unsupported
    // GEMM/IBP/Conv routes typed-refuse and keep their established CPU paths.
    let gemm_engine = compute_device.as_deref().map(|d| d as &dyn GemmEngine);

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

    // For graph models, automatically disable features not yet supported.
    // ReLU splitting has cut plumbing; graph input splitting does not.
    let is_input_split_for_graph = effective_branching_heuristic
        .as_ref()
        .map(|h| matches!(h, BranchingHeuristic::InputSplit))
        .unwrap_or(false);
    let (use_alpha_effective, crown_ibp_effective, cuts_supported) =
        if is_graph_model && is_input_split_for_graph {
            // Input splitting: α-CROWN supported (tighter initial bounds via CROWN-IBP
            // intermediates in collect_alpha_crown_bounds_dag, #3357); crown_ibp and cuts
            // still not supported for input splitting.
            if crown_ibp && !json {
                eprintln!(
                    "Note: Graph model with input splitting - disabling unsupported \
                 CROWN-IBP (crown_ibp={crown_ibp})"
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
    let preset_enables_cuts = preset_config
        .as_ref()
        .and_then(|p| p.bab.cuts.enabled)
        .unwrap_or(false);
    validate_cut_request(
        enable_cuts || (preset_enables_cuts && !no_cuts),
        cuts_supported,
    )?;

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
        let eff_enable_cuts =
            resolve_enable_cuts(preset_enables_cuts, enable_cuts, no_cuts, cuts_supported);
        let eff_clip_interm_domain = clip_interm_domain
            || preset_config
                .as_ref()
                .and_then(|p| p.bab.clip.interm_domain)
                .unwrap_or(false);
        // `interm_transfer` falls back to the BetaCrownConfig default, not `false`.
        let eff_interm_transfer =
            effective_interm_transfer(interm_transfer, preset_config.as_ref());
        let eff_max_queue_bytes = max_queue_bytes
            .or_else(|| {
                preset_config
                    .as_ref()
                    .and_then(|preset| preset.bab.max_queue_bytes)
            })
            .unwrap_or_else(|| BetaCrownConfig::default().max_queue_bytes);

        // Display effective values (from CLI, preset, or defaults)
        let branching_str = branching.as_deref().unwrap_or("(preset/default)");
        println!(
            "Config: max_domains={}, max_queue_bytes={}, timeout={}s, max_depth={}, branching={}, fsb_candidates={}, use_alpha={}, alpha_iter={}, alpha_grad={}, alpha_opt={}, alpha_lr={}, crown_ibp_interm={}, beta_iter={}, lr_beta={}, crown_ibp={}, batch_size={}, parallel={}, enable_cuts={}, max_cuts={}, min_cut_depth={}, biccos_strengthen={}, biccos_drop_ratio={}, relaxed_clip={}, relaxed_clip_iters={}, clip_interm_domain={}, clip_interm_topk={}, clip_in_alpha_crown={}, clip_interm_prune={}, clip_interm_final_layer={}, interm_transfer={}, pgd_attack={}",
            max_domains.map(|v| v.to_string()).unwrap_or_else(|| "(preset/default)".to_string()),
            eff_max_queue_bytes,
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
    if let Some(resolved) = auto_branching.as_ref() {
        branching::apply_resolved_auto_branching_runtime_policy(&mut config, resolved);
    }
    apply_vgg_abcrown_bound_mode(&mut config, vgg_abcrown_decision);
    apply_instance_overrides(&mut config, &instance_overrides);
    apply_root_post_c_survivor_env(&mut config);

    // CLI flags override preset values ONLY when explicitly provided.
    // For Option<T> fields: only override when Some (CLI explicitly set).
    // For bool flags that default to false: only override when true (CLI explicitly
    // enables). When false (the default), preserve the preset value. This prevents
    // CLI defaults from silently overwriting preset-enabled features like relaxed_clip,
    // clip_interm_domain, etc. Bool flags that default to TRUE (currently
    // pgd_attack) cannot use this rule — their `true` is indistinguishable from
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
    if let Some(v) = max_queue_bytes {
        config.max_queue_bytes = v;
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
    // Preserve supported-model CLI/preset cut requests long enough for
    // BetaCrownConfig::validate() to return the quarantine error below.
    // An explicit unsupported-model CLI request was rejected above; a preset
    // request is conservatively forced off for that model class.
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
    config.phase_budget =
        effective_engine_phase_budget(config.enable_pgd_attack, config.phase_budget);
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
    // Intermediate-bounds mode: CLI-only override. `--crown-ibp-intermediates`
    // is a bare flag, so `false` cannot be distinguished from "not passed" —
    // assigning unconditionally therefore CLOBBERED whatever
    // `solver/bab.alpha_crown.fix_interm_bounds` a preset had just applied,
    // which is why the `ny vnncomp` path (which passes `false` positionally)
    // could never reach the tighter CROWN-IBP-intermediate mode. Applying it
    // only when the flag was actually passed keeps `--crown-ibp-intermediates`
    // winning over the preset and leaves preset-free runs byte-identical
    // (`AlphaCrownConfig::default().fix_interm_bounds == true`).
    if crown_ibp_intermediates {
        config.alpha_config.fix_interm_bounds = false;
    }
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
    config.verification_artifact_authority = proof_opts.verification_artifact_authority();
    config.validate()?;
    if complete_verifier == CompleteVerifierArg::Mip && config.timeout.is_zero() {
        anyhow::bail!("unbounded MIP verification is unsupported; pass a positive --timeout");
    }

    // Preset-scoped per-node CROWN-IBP time budget (#4413, #cgan-bn11-budget):
    // stamp the loaded graph so every downstream clone (dispatch, disjunctive
    // per-disjunct graphs, engine-configured copies) inherits the policy.
    // Sequential verifiers instead pass the same config directly into each
    // root/child collection. With no explicit overrides, both retain the 2 s
    // floor and derive the cap adaptively from the remaining collection budget.
    if let BetaCrownModel::Graph(graph) = &mut model_net {
        graph.set_crown_ibp_per_node_time_budget(config.crown_ibp_per_node_time_budget());
    }

    // Auto-promote gpu_bab for graph models with supported heuristics on
    // single-objective specs. The DomainList frontier is non-regressing and
    // faster than the heap path for these lanes (#4406). Generic disjunctive
    // multi-clause specs remain excluded (#4409); the canonical two-singleton
    // InputSplit shape is promoted for its explicit sequential scheduler only
    // when the preset opts in (currently LinearizeNN 2024).
    let gpu_bab = should_auto_promote_gpu_bab(
        gpu_bab,
        is_graph_model,
        &config.branching_heuristic,
        config.input_split_independent_singleton_disjunction,
        vnnlib_spec.as_ref(),
    );

    let is_mip_only = complete_verifier == CompleteVerifierArg::Mip
        && matches!(&model_net, BetaCrownModel::Sequential(_));
    let run_upfront_pgd = effective_upfront_pgd_with_vgg(
        config.enable_pgd_attack,
        preset_config.as_ref(),
        vgg_abcrown_decision,
    );
    // Relational dispatch can convert a Sequential multi-row conjunction to the
    // Graph verifier after frontend routing has otherwise finished. Resolve that
    // representation/config boundary here as evidence-only planning too, so the
    // emitted treatment describes the verifier that will actually own BaB. MIP-
    // only Sequential invocations bypass this Graph route and retain their
    // original treatment.
    let late_sequential_conjunction_graph_config = planned_late_sequential_conjunction_graph_config(
        &model_net,
        &config,
        vnnlib_spec.as_ref(),
        use_relu_split,
        is_mip_only,
    );
    let treatment_config = late_sequential_conjunction_graph_config
        .as_ref()
        .unwrap_or(&config);
    let treatment_model_is_graph =
        is_graph_model || late_sequential_conjunction_graph_config.is_some();
    // `verify_standard` is the sole direct route with an engine-owned deferred
    // PGD fallback. Relational Sequential dispatch gates both CLI attack sites
    // on `run_upfront_pgd`, while Graph, late-Graph, and MIP-only routes have no
    // internal consumer and therefore require the explicit outer wrapper.
    let direct_internal_deferred_pgd_consumer_available =
        matches!(&model_net, BetaCrownModel::Sequential(_))
            && !is_mip_only
            && late_sequential_conjunction_graph_config.is_none()
            && !dispatch::routes_to_relational_verifier(vnnlib_spec.as_ref(), has_relational);
    let deferred_pgd_schedule =
        config.enable_pgd_attack && preset_uses_deferred_pgd_schedule(preset_config.as_ref());
    // Prefer the explicitly planned outer VNN-COMP phase when available. This
    // gives exactly one owner: direct standard Sequential uses its engine
    // fallback only when no outer route exists; every other admitted route is
    // outer-owned.
    let deferred_pgd_owner = resolve_deferred_pgd_owner(
        deferred_pgd_schedule,
        direct_internal_deferred_pgd_consumer_available,
        deferred_pgd_consumer_available,
    );
    let outer_wrapper_owns_deferred_pgd = deferred_pgd_owner == DeferredPgdOwner::OuterWrapper;
    let engine_owns_deferred_pgd = deferred_pgd_owner == DeferredPgdOwner::InternalEngine;
    require_deferred_pgd_consumer(
        preset_config.as_ref(),
        config.enable_pgd_attack,
        is_graph_model,
        late_sequential_conjunction_graph_config.is_some(),
        is_mip_only,
        direct_internal_deferred_pgd_consumer_available,
        deferred_pgd_consumer_available,
    )?;
    // Build the promotional-evidence projection only after every configuration
    // and frontend route decision has resolved. Keep it owned: `config.timeout`
    // is rebased below after exact-CNF work, while these treatment fields must
    // describe one stable invocation at every verdict-emission seam.
    let effective_treatment = output::EffectiveTreatmentProjection::from_resolved(
        treatment_config,
        treatment_model_is_graph,
        late_sequential_conjunction_graph_config.is_some(),
        use_relu_split,
        gpu_bab,
        run_upfront_pgd,
        vgg_abcrown_decision.is_some(),
        complete_verifier,
        requested_backend,
        effective_backend.to_string(),
        vnnlib_spec.as_ref().map(|spec| spec.is_disjunction),
    )
    .with_deferred_pgd_schedule(deferred_pgd_schedule)
    .with_terminal_peel(peel_last_softmax_layer, applied_terminal_peel)
    .with_backend_receipt(&proof_backend_receipt);
    let verification_start = std::time::Instant::now();
    let overall_deadline = resolve_overall_deadline(
        verification_start,
        config.timeout,
        instance_overrides.authoritative_deadline,
    )?;
    let pre_dispatch_internal_deadline = resolve_internal_authority_deadline(
        overall_deadline,
        outer_wrapper_owns_deferred_pgd,
        instance_overrides.outer_deferred_internal_deadline,
    )?;

    // CNF-recovery driver: a
    // SAT-encoded ReLU gadget (sat_relu) is decompiled to its source CNF and
    // decided EXACTLY by the in-process `ay-sat` CDCL solver — a replayed
    // resolution-DAG certificate on UNSAT, an in-process-confirmed boolean
    // witness on SAT (re-confirmed downstream by the vnncomp ONNX-Runtime
    // trusted-oracle gate). Placed BEFORE the MIP/BaB fork: bound propagation is
    // vacuous on this family by construction and the float-MIP fallback carries
    // no certificate. Fail-closed bit-exact detection: non-gadget models fall
    // through with zero behavior change. Disable with NY_NO_CNF_ROUTE=1.
    if let Some(vnnlib) = vnnlib_spec.as_ref() {
        if let Some(result) = cnf_route::try_cnf_recovery(
            &model_net,
            input.shape(),
            vnnlib,
            pre_dispatch_internal_deadline,
        ) {
            // Recheck at the final caller-side publication seam. The CNF
            // route gates after proof/witness work, but formatting may still
            // cross the authoritative deadline by a narrow scheduler race.
            let result = dispatch::gate_result_at_deadline(
                result,
                verification_start,
                pre_dispatch_internal_deadline,
            );
            output::output_result(
                &result,
                &property,
                Some(model.as_path()),
                epsilon,
                effective_threshold,
                verify_upper,
                json,
                sigmoid_peeled,
                &effective_treatment,
            )?;
            return Ok(());
        }
    }

    if let Some(deadline) = pre_dispatch_internal_deadline {
        let now = std::time::Instant::now();
        if now >= deadline {
            let result = BetaCrownResult {
                result: BabVerificationStatus::Timeout,
                domains_explored: 0,
                domains_verified: 0,
                cuts_generated: 0,
                max_depth_reached: 0,
                time_elapsed: verification_start.elapsed(),
                output_bounds: None,
            };
            output::output_result(
                &result,
                &property,
                Some(model.as_path()),
                epsilon,
                effective_threshold,
                verify_upper,
                json,
                sigmoid_peeled,
                &effective_treatment,
            )?;
            return Ok(());
        }
        // Charge time spent in exact-CNF qualification/solving/certification
        // to the same authoritative timeout as the fallback. Dispatch ledgers
        // consume this exact sub-second duration and cannot reset the budget.
        config.timeout = deadline.duration_since(now);
    } else {
        // Low-level BaB APIs still carry a concrete `Duration`. Preserve the
        // CLI's logical no-deadline contract with a representable,
        // decades-long engine horizon rather than an overflowing `u64::MAX`
        // sentinel.
        config.timeout = verify::phase_budget::operational_unbounded_timeout();
    }

    let mut dispatch_context = DispatchContext {
        model_path: &model,
        onnx_load_config: &onnx_load_config,
        model_net: &mut model_net,
        input: &input,
        config: &config,
        effective_treatment: &effective_treatment,
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
        engine_owns_deferred_pgd,
        outer_wrapper_owns_deferred_pgd,
        safenlp_direct_mip_first: instance_overrides.safenlp_direct_mip_first,
        cgan_input_leaf_route: instance_overrides.cgan_input_leaf_route,
        gemm_engine,
        attack_engine_source,
        compute_device: compute_device.clone(),
        allow_heuristic_logsoftmax,
        allow_heuristic_softmax,
        input_split_metrics_jsonl: input_split_metrics_jsonl.as_deref(),
        domain_batch_metrics_jsonl: domain_batch_metrics_jsonl.as_deref(),
        verification_start,
        overall_deadline: pre_dispatch_internal_deadline,
        post_bab_wrapper_attack_enabled: instance_overrides.post_bab_wrapper_attack_enabled,
        json,
        sigmoid_peeled,
        proof_opts: &proof_opts,
    };

    if is_mip_only {
        return run_mip_only(&dispatch_context, mip_solver);
    }

    run_bab_with_fallback(&mut dispatch_context, complete_verifier, mip_solver)
}

/// Plan the late Sequential-conjunction Graph representation used by BaB.
///
/// This is intentionally side-effect free: the real conversion still happens
/// at verification dispatch, while treatment evidence can resolve the same
/// config before any early verdict-emission seam.
fn planned_late_sequential_conjunction_graph_config(
    model: &BetaCrownModel,
    config: &BetaCrownConfig,
    vnnlib: Option<&ny_onnx::vnnlib::VnnLibSpec>,
    use_relu_split: bool,
    is_mip_only: bool,
) -> Option<BetaCrownConfig> {
    if is_mip_only || !matches!(model, BetaCrownModel::Sequential(_)) {
        return None;
    }
    vnnlib.and_then(|spec| {
        verify::planned_sequential_conjunction_graph_config(config, spec, use_relu_split)
    })
}

/// Whether a disjunctive VNN-LIB spec can be soundly decomposed into exactly
/// two independent singleton DomainList searches.
///
/// Keep this gate deliberately structural.  In particular, every clause must
/// share the root input box and the canonical flat constraint list must agree
/// exactly with the clause representation.  Any richer or malformed shape
/// remains on the existing grouped CPU lane.
pub(in crate::commands::beta_crown) fn supports_independent_singleton_domain_list_spec(
    spec: &ny_onnx::vnnlib::VnnLibSpec,
) -> bool {
    if !spec.is_disjunction
        || spec.dual_network.is_some()
        || spec.output_constraint_clauses.len() != 2
        || spec
            .output_constraint_clauses
            .iter()
            .any(|clause| clause.len() != 1)
    {
        return false;
    }

    if !spec.per_clause_input_bounds.is_empty()
        && (spec.per_clause_input_bounds.len() != 2
            || spec
                .per_clause_input_bounds
                .iter()
                .any(|bounds| !bounds.is_empty()))
    {
        return false;
    }

    let canonical_flat: Vec<_> = spec
        .output_constraint_clauses
        .iter()
        .flatten()
        .cloned()
        .collect();
    spec.output_constraints == canonical_flat
}

/// Determine whether to auto-promote to the DomainList (`gpu_bab`) path.
///
/// Returns `true` when:
/// - `gpu_bab` was explicitly requested (CLI `--gpu-bab`), OR
/// - The model is a graph, the branching heuristic is supported
///   (`BoundImpact` or `InputSplit`), and the spec is not a multi-clause
///   disjunction, OR
/// - The model is a graph using `InputSplit` and the spec is the narrowly
///   supported exactly-two-singleton disjunction above, with the typed preset
///   opt-in enabled. The opt-in defaults to false and is currently armed only
///   by the LinearizeNN 2024 competition preset.
///
/// Part of #4406: DomainList dispatch promotion for single-objective lanes.
/// Re: #2326 — promotion does not resolve queue/cache hardening.
pub(crate) fn should_auto_promote_gpu_bab(
    cli_gpu_bab: bool,
    is_graph_model: bool,
    heuristic: &BranchingHeuristic,
    independent_singleton_opt_in: bool,
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
    // Guard: generic multi-clause disjunctive specs regress on gpu_bab (#4409).
    // The sole opt-in exception is decomposed explicitly by the prechecked
    // verifier: two singleton clauses over one canonical input box,
    // InputSplit only. The typed config default is false so unrelated
    // categories (notably cGAN and ml4acopf) cannot enter this route.
    let is_multi_clause_disjunction = vnnlib_spec
        .is_some_and(|spec| spec.is_disjunction && spec.output_constraint_clauses.len() > 1);
    if !is_multi_clause_disjunction {
        return true;
    }

    independent_singleton_opt_in
        && matches!(heuristic, BranchingHeuristic::InputSplit)
        && vnnlib_spec.is_some_and(supports_independent_singleton_domain_list_spec)
}

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

/// Parse a preset's MIP solver into the first-party solver policy.
///
/// Legacy third-party solver names map to ay with a warning so historical
/// category presets remain usable. Unknown values are a hard preset error.
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

/// Reject requested cuts when the selected model/split path cannot represent
/// them. CLI and preset requests must never be silently ignored.
fn validate_cut_request(cuts_requested: bool, cuts_supported: bool) -> Result<()> {
    if cuts_requested && !cuts_supported {
        anyhow::bail!(
            "--enable-cuts / preset bab.cuts.enabled is unsupported for graph \
             input splitting; choose ReLU splitting or disable cuts"
        );
    }
    Ok(())
}

/// Resolve the requested `enable_cuts`: explicit CLI enable/disable wins over
/// the preset value already applied to the config. Unsupported-model requests
/// have already been rejected by [`validate_cut_request`], unless an explicit
/// `--no-cuts` neutralized the preset/CLI request. The resulting supported-model
/// request is still subject to the certificate-authority quarantine in
/// `BetaCrownConfig::validate()`.
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
