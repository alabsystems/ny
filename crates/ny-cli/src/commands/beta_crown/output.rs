// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
#[cfg(feature = "mip")]
use ny_core::{Bound, VerificationResult};
use ny_onnx::vnnlib::VnnLibSpec;
use ny_propagate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownResult, BranchingHeuristic, ConvMode,
    DepthTwoBranchLookaheadMode, GradientMethod, KfsbReduceOp, Optimizer,
};
use serde::Serialize;
use std::path::{Path, PathBuf};
#[cfg(feature = "mip")]
use std::time::Duration;

use super::AppliedTerminalPeel;
use crate::commands::backend::ProofBackendReceipt;
use crate::commands::verify::{exit_codes, json_f32};
use crate::{BackendArg, CompleteVerifierArg};

use std::cell::{Cell, RefCell};

/// A solver ingress that owns the terminal result of an in-process
/// `vnncomp` verification attempt.
///
/// This is capture metadata, not a verdict.  The marked SafeNLP route is
/// deliberately terminal even when its sound result is `timeout`/`unknown`:
/// once admitted, the caller may not enter any downstream post-BaB attack or
/// second-solver lane.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum CapturedTerminalIngress {
    #[default]
    None,
    #[cfg_attr(not(feature = "mip"), allow(dead_code))]
    RequireSafeNlpMarkedMarginSharedBinaryPrefix,
}

thread_local! {
    /// In-process capture sink for the competition JSON verdict.
    ///
    /// When `Some`, the `--json` output sites (`output_result`, the SMT path, and
    /// the HiGHS-MIP path) store the rendered competition JSON string here instead
    /// of printing it to stdout, and they SKIP the `std::process::exit(...)` that a
    /// non-verified verdict would normally trigger. This lets the native `vnncomp`
    /// subcommand call `handle_beta_crown_command` directly (no shell-out, no second
    /// `ny` process), capture the exact same verdict JSON the shell wrapper used to
    /// parse, and translate it into the VNN-COMP result string in-process.
    ///
    /// SOUNDNESS: the captured string is byte-for-byte the same JSON the CLI would
    /// have printed in `--json` mode — the verdict mapping is unchanged. Suppressing
    /// the process exit is purely a control-flow concern (the exit code is a
    /// CLI-ergonomics signal, not part of the soundness contract; the VNN-COMP result
    /// is carried entirely by the JSON `status`/`counterexample_vnnlib` fields).
    static CAPTURE_SINK: RefCell<Option<String>> = const { RefCell::new(None) };

    /// Caller-local terminal-ingress provenance for the same capture session.
    ///
    /// The marker is set from the typed MIP ingress, after its backend/seed
    /// invariants are admitted.  In particular, it is not inferred later from
    /// ambient environment variables, category names, or log text.
    static CAPTURE_TERMINAL_INGRESS: Cell<CapturedTerminalIngress> =
        const { Cell::new(CapturedTerminalIngress::None) };
}

/// Begin capturing the competition JSON verdict on this thread.
///
/// Any previously captured value is cleared. Call [`take_captured_json`] afterwards
/// to retrieve the rendered verdict, and [`end_capture`] (or the returned value's
/// drop) to stop capturing.
pub(crate) fn begin_capture() {
    CAPTURE_SINK.with(|sink| *sink.borrow_mut() = Some(String::new()));
    CAPTURE_TERMINAL_INGRESS.with(|ingress| ingress.set(CapturedTerminalIngress::None));
}

/// Stop capturing on this thread, discarding any buffered verdict.
pub(crate) fn end_capture() {
    CAPTURE_SINK.with(|sink| *sink.borrow_mut() = None);
    CAPTURE_TERMINAL_INGRESS.with(|ingress| ingress.set(CapturedTerminalIngress::None));
}

/// Take the captured competition JSON verdict, if capture is active and a verdict
/// was rendered. Returns `None` when not capturing or when no verdict was produced.
pub(crate) fn take_captured_json() -> Option<String> {
    CAPTURE_SINK.with(|sink| {
        let mut guard = sink.borrow_mut();
        match guard.as_mut() {
            Some(buf) if !buf.is_empty() => Some(std::mem::take(buf)),
            _ => None,
        }
    })
}

/// Mark the active capture as owned by the admitted marked-margin SafeNLP
/// ingress.  Outside an in-process capture session this is intentionally inert:
/// the standalone `beta-crown` command already returns directly to its caller.
#[cfg(feature = "mip")]
pub(super) fn mark_captured_safenlp_marked_margin_terminal() {
    if is_capturing() {
        CAPTURE_TERMINAL_INGRESS.with(|ingress| {
            ingress.set(CapturedTerminalIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix);
        });
    }
}

/// Take terminal-ingress provenance for the active in-process capture.
///
/// Like [`take_captured_json`], this does not end the capture session; callers
/// still invoke [`end_capture`] so every thread-local field is reset together.
pub(crate) fn take_captured_terminal_ingress() -> CapturedTerminalIngress {
    CAPTURE_TERMINAL_INGRESS.with(Cell::take)
}

/// Returns `true` when the competition JSON verdict should be captured rather than
/// printed (and the process-exit suppressed). Used by the MIP/SMT `--json` output
/// sites to decide whether to skip `std::process::exit`. Only referenced from the
/// `mip`-gated paths, so it is dead code in a default (non-mip) build.
#[cfg_attr(not(feature = "mip"), allow(dead_code))]
pub(super) fn is_capturing() -> bool {
    CAPTURE_SINK.with(|sink| sink.borrow().is_some())
}

/// Route a rendered competition JSON verdict to the active capture sink (if any) or
/// to stdout. Returns `true` when the verdict was captured (and the caller MUST NOT
/// `std::process::exit`), `false` when it was printed normally.
pub(super) fn emit_competition_json(json: &str) -> bool {
    CAPTURE_SINK.with(|sink| {
        let mut guard = sink.borrow_mut();
        if let Some(buf) = guard.as_mut() {
            *buf = json.to_string();
            true
        } else {
            println!("{json}");
            false
        }
    })
}

const EFFECTIVE_TREATMENT_SCHEMA: &str = "ny_beta_crown_effective_treatment_v1";

/// Stable, deliberately narrow projection of the resolved β-CROWN treatment.
///
/// This is built only after preset, model-aware auto-routing, instance, env, and
/// CLI precedence have all been applied. It intentionally does not serialize the
/// full pre-1.0 `BetaCrownConfig`: that surface is large, unstable, and still
/// omits frontend route decisions such as graph/ReLU dispatch and the quarantined
/// proof backend. The projection contains the knobs needed to audit promotional
/// alpha-beta-CROWN transfer experiments and their known runtime gate confounds.
#[derive(Debug, Clone, Serialize)]
pub(super) struct EffectiveTreatmentProjection {
    schema: &'static str,
    batch: EffectiveBatchProjection,
    branching: EffectiveBranchingProjection,
    attack: EffectiveAttackProjection,
    alpha_crown: EffectiveAlphaProjection,
    beta_crown: EffectiveBetaProjection,
    clip: EffectiveClipProjection,
    root: EffectiveRootProjection,
    invprop: EffectiveInvpropProjection,
    softmax: EffectiveSoftmaxProjection,
    route: EffectiveRouteProjection,
}

#[derive(Debug, Clone, Serialize)]
struct EffectiveBatchProjection {
    configured_size: usize,
    build_batch_size: Option<usize>,
    auto_enlarge: bool,
    adaptive_microbatch_controller_armed: bool,
    max_relu_split_depth: usize,
    min_fill_ratio: f32,
    parallel_children: bool,
}

#[derive(Debug, Clone, Serialize)]
struct EffectiveBranchingProjection {
    heuristic: &'static str,
    input_split_coeff_threshold: f32,
    input_split_adv_check: i32,
    reorder_bab: bool,
    configured_candidates: usize,
    effective_wave_candidates: usize,
    configured_reduce_op: &'static str,
    effective_wave_reduce_op: &'static str,
    kfsb_multi_configured: bool,
    kfsb_cert_reuse_configured: bool,
    kfsb_cert_reuse_armed: bool,
    multi_objective_critical_kfsb_configured: bool,
    kfsb_multi_env_override: Option<bool>,
    depth_two_lookahead_mode: &'static str,
    depth_two_lookahead_candidates: usize,
    depth_two_lookahead_top_rounds: usize,
    depth_two_lookahead_discount: f64,
    depth_two_lookahead_round_zero_supported: bool,
    adaptive_depth_shadow_env_armed: bool,
    adaptive_depth_select_env_armed: bool,
    adaptive_depth_commit_env_armed: bool,
    depth_two_lookahead_legacy_observer_conflict: bool,
    wave_kfsb_armed: bool,
    scorer_fix_env_armed: bool,
    competing_branch_experiment_armed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct EffectiveAttackProjection {
    enabled: bool,
    schedule: &'static str,
    pgd_restarts: usize,
    pgd_steps: usize,
}

#[derive(Debug, Clone, Serialize)]
struct EffectiveAlphaProjection {
    enabled: bool,
    iterations: usize,
    learning_rate: f32,
    lr_decay: f32,
    optimizer: &'static str,
    gradient_method: &'static str,
    fix_interm_bounds: bool,
    cgan_sparse_target_complete_root: bool,
    cgan_complete_crown_ibp_root: bool,
    full_conv_alpha: bool,
    adaptive_skip: bool,
    adaptive_skip_depth_threshold: usize,
    early_stop_patience: usize,
    start_save_best: f32,
    final_bound_only_env_armed: bool,
    packed_graph_alpha_queue_env_armed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct EffectiveBetaProjection {
    iterations: usize,
    learning_rate_alpha: f32,
    learning_rate_beta: f32,
    max_depth: usize,
    early_stop_patience: usize,
}

#[derive(Debug, Clone, Serialize)]
struct EffectiveClipProjection {
    interm_domain: bool,
    interm_topk: usize,
    in_alpha_crown: bool,
    prune: bool,
    use_final_layer: bool,
    /// Typed request only. Runtime execution belongs in the sibling
    /// `execution_observations` object and is never inferred from this bit.
    input_split_fresh_domain_clip_configured: bool,
}

#[derive(Debug, Clone, Serialize)]
struct EffectiveRootProjection {
    dense_head_configured: bool,
    sparse_configured: bool,
    sparse_gate: &'static str,
    sparse_effective_armed: bool,
    atomic_root_c_margin_iterations: usize,
    /// Exact `NY_ROOT_SPEC_PRUNE=1` request. Actual planning/application is
    /// reported only by the execution-observations sibling.
    root_spec_prune_requested: bool,
}

/// Resolved INVPROP configuration plus metadata for the matrix attached to the
/// top-level invocation. This intentionally records configuration and ingress
/// state, not runtime activation: serial disjunctive verification may later
/// construct a clause-local matrix after this projection has been sealed.
#[derive(Debug, Clone, Serialize)]
struct EffectiveInvpropProjection {
    enabled: bool,
    apply_output_constraints_to: Vec<String>,
    tighten_input_bounds: bool,
    best_of_oc_and_no_oc: bool,
    directly_optimize: Vec<String>,
    share_gammas: bool,
    per_layer_gammas: bool,
    optimize_gammas: bool,
    gamma_lr: f32,
    top_level_output_constraint_matrix: Option<EffectiveOutputConstraintMatrixProjection>,
    serial_clause_rebinding: &'static str,
    split_lift_requested: bool,
    split_lift_effective_armed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct EffectiveOutputConstraintMatrixProjection {
    rows: usize,
    columns: usize,
    rhs_entries: usize,
    is_conjunction: bool,
    clause_indices_present: bool,
    clause_count: Option<usize>,
    clause_row_counts: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize)]
struct EffectiveSoftmaxProjection {
    /// Exact `NY_SOFTMAX_OBJECTIVE_ENVELOPE=1` process treatment. This reports
    /// arming, not whether the measured graph actually traversed a Softmax.
    objective_envelope_env_armed: bool,
    /// Frontend request after CLI and typed in-process VNN-COMP precedence.
    terminal_peel_requested: bool,
    /// The fail-closed joint model/property pass actually removed a terminal
    /// activation and rewrote every present output atom.
    terminal_peel_applied: bool,
    /// Exact activation removed.  `none` is the only value when
    /// `terminal_peel_applied=false`.
    terminal_peel_activation: AppliedTerminalPeel,
}

#[derive(Debug, Clone, Serialize)]
struct EffectiveRouteProjection {
    model_kind: &'static str,
    configured_conv_mode: &'static str,
    vgg_abcrown_treatment_active: bool,
    late_sequential_conjunction_graph_upgrade: bool,
    use_relu_split: bool,
    gpu_bab: bool,
    run_upfront_pgd: bool,
    complete_verifier: &'static str,
    /// Compatibility name for the backend selected after CLI/preset/AUTO
    /// precedence and before live proof qualification.
    selected_backend: String,
    requested_backend: String,
    backend_request_source: &'static str,
    backend_selection_reason: Option<String>,
    effective_backend: String,
    wgpu_qualification: &'static str,
    wgpu_qualification_provenance: Option<String>,
    wgpu_failed_rung: Option<String>,
    backend_fallback_reason: Option<String>,
    /// Compatibility backend-kind field retained for existing evidence readers.
    proof_backend: String,
    proof_backend_provenance: String,
    property_is_disjunction: Option<bool>,
    intermediate_bound_transfer: bool,
    use_forward_bounds: bool,
    use_crown_ibp: bool,
    input_split_ibp_enhancement: bool,
    /// Typed request before route/property/proof-authority admission.
    input_split_conic_objective_configured: bool,
    /// Resolved frontend eligibility. Actual row admission remains subject to
    /// the bit-exact objective-shape detector at the proof handoff.
    input_split_conic_objective_route_eligible: bool,
}

#[derive(Debug, Clone, Copy)]
struct RuntimeTreatmentGates {
    adaptive_microbatch_controller: bool,
    kfsb_multi_env_override: Option<bool>,
    kfsb_cert_reuse_armed: bool,
    wave_candidates: usize,
    wave_reduce_op: KfsbReduceOp,
    scorer_fix: bool,
    competing_branch_experiment: bool,
    final_alpha_bound_only: bool,
    packed_graph_alpha_queue: bool,
    root_sparse_gate: &'static str,
    root_sparse_effective_armed: bool,
    root_spec_prune_requested: bool,
    adaptive_depth_shadow_env_armed: bool,
    adaptive_depth_select_env_armed: bool,
    adaptive_depth_commit_env_armed: bool,
    depth_two_lookahead_legacy_observer_conflict: bool,
    invprop_split_lift_requested: bool,
    softmax_objective_envelope: bool,
}

impl RuntimeTreatmentGates {
    fn capture(config: &BetaCrownConfig) -> Self {
        let exactly_one = |name: &str| std::env::var(name).ok().as_deref() == Some("1");
        let truthy = |name: &str| {
            std::env::var(name)
                .ok()
                .map(|value| {
                    let value = value.trim();
                    value == "1"
                        || value.eq_ignore_ascii_case("true")
                        || value.eq_ignore_ascii_case("on")
                })
                .unwrap_or(false)
        };
        let childsim = exactly_one("NY_BRANCH_KFSB_CHILDSIM");
        let kfsb_multi_env_override = match std::env::var("NY_MO_KFSB").ok().as_deref() {
            Some("1") => Some(true),
            Some("0") => Some(false),
            _ if childsim => Some(true),
            _ => None,
        };
        let wave_candidates = std::env::var("NY_MO_KFSB_K")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(if childsim { 2 } else { config.fsb_candidates });
        let wave_reduce_op = match std::env::var("NY_MO_KFSB_REDUCE").ok().as_deref() {
            Some("max") => KfsbReduceOp::Max,
            Some("min") => KfsbReduceOp::Min,
            _ => config.kfsb_reduce_op,
        };
        let root_sparse_gate = match std::env::var_os("NY_ROOT_SPARSE_INTERM_CROWN") {
            None => "absent",
            Some(value) if value == "1" => "enable",
            Some(value) if value == "0" => "disable",
            Some(_) => "invalid",
        };
        let root_sparse_effective_armed = match root_sparse_gate {
            "absent" => config.root_sparse_interm_crown,
            "enable" => true,
            "disable" | "invalid" => false,
            _ => unreachable!("root sparse gate is a closed enum"),
        };
        let adaptive_depth_shadow_env_armed = exactly_one("NY_MO_ADAPTIVE_DEPTH_SHADOW");
        let adaptive_depth_select_env_armed = exactly_one("NY_MO_ADAPTIVE_DEPTH_SELECT");
        let adaptive_depth_commit_env_armed = exactly_one("NY_MO_ADAPTIVE_DEPTH_COMMIT");
        let depth_two_lookahead_legacy_observer_conflict = adaptive_depth_shadow_env_armed
            || adaptive_depth_select_env_armed
            || adaptive_depth_commit_env_armed
            || exactly_one("NY_MO_KFSB_F64_SHADOW");

        Self {
            adaptive_microbatch_controller: exactly_one("NY_ADAPTIVE_MICROBATCH_CONTROLLER"),
            kfsb_multi_env_override,
            kfsb_cert_reuse_armed: config.kfsb_cert_reuse_armed(),
            wave_candidates,
            wave_reduce_op,
            scorer_fix: exactly_one("NY_MO_SCORER_FIX"),
            competing_branch_experiment: exactly_one("NY_BRANCH_LA")
                || exactly_one("NY_BRANCH_LA_PROBE")
                || exactly_one("NY_BRANCH_STEM"),
            final_alpha_bound_only: exactly_one("NY_ALPHA_FINAL_BOUND_ONLY"),
            packed_graph_alpha_queue: exactly_one("NY_PACKED_GRAPH_ALPHA_QUEUE"),
            root_sparse_gate,
            root_sparse_effective_armed,
            root_spec_prune_requested: exactly_one("NY_ROOT_SPEC_PRUNE"),
            adaptive_depth_shadow_env_armed,
            adaptive_depth_select_env_armed,
            adaptive_depth_commit_env_armed,
            depth_two_lookahead_legacy_observer_conflict,
            invprop_split_lift_requested: truthy("NY_INVPROP_SPLIT_LIFT"),
            softmax_objective_envelope: exactly_one("NY_SOFTMAX_OBJECTIVE_ENVELOPE"),
        }
    }
}

impl EffectiveTreatmentProjection {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_resolved(
        config: &BetaCrownConfig,
        is_graph_model: bool,
        late_sequential_conjunction_graph_upgrade: bool,
        use_relu_split: bool,
        gpu_bab: bool,
        run_upfront_pgd: bool,
        vgg_abcrown_treatment_active: bool,
        complete_verifier: CompleteVerifierArg,
        selected_backend: BackendArg,
        proof_backend: impl Into<String>,
        property_is_disjunction: Option<bool>,
    ) -> Self {
        Self::from_resolved_with_gates(
            config,
            is_graph_model,
            late_sequential_conjunction_graph_upgrade,
            use_relu_split,
            gpu_bab,
            run_upfront_pgd,
            vgg_abcrown_treatment_active,
            complete_verifier,
            selected_backend,
            proof_backend.into(),
            property_is_disjunction,
            RuntimeTreatmentGates::capture(config),
        )
    }

    /// Attach the frontend terminal-peel receipt after model loading. Keeping
    /// this separate from propagation configuration prevents a loader decision
    /// from masquerading as an engine knob while still sealing it in every
    /// captured competition JSON verdict.
    pub(super) fn with_terminal_peel(
        mut self,
        requested: bool,
        applied: AppliedTerminalPeel,
    ) -> Self {
        self.softmax.terminal_peel_requested = requested;
        self.softmax.terminal_peel_applied = applied.applied();
        self.softmax.terminal_peel_activation = applied;
        self
    }

    /// Correct the non-upfront attack projection for reference `pgd_order:
    /// after`. A false `run_upfront_pgd` used to be reported generically as
    /// `input_bab`, hiding that this schedule is deferred until after BaB.
    pub(super) fn with_deferred_pgd_schedule(mut self, deferred: bool) -> Self {
        if deferred && self.attack.enabled && !self.route.run_upfront_pgd {
            self.attack.schedule = "deferred";
        }
        self
    }

    /// Replace the legacy backend pair with the full runtime decision receipt.
    ///
    /// The base constructor remains intentionally usable by the many synthetic
    /// treatment tests that do not execute backend qualification. Production
    /// command paths must call this after constructing their proof device.
    pub(super) fn with_backend_receipt(mut self, receipt: &ProofBackendReceipt) -> Self {
        self.route.selected_backend = receipt.requested.to_string();
        self.route.requested_backend = receipt.requested.to_string();
        self.route.backend_request_source = receipt.request_source.as_str();
        self.route.backend_selection_reason = receipt.selection_reason.clone();
        self.route.effective_backend = receipt.effective.to_string();
        self.route.wgpu_qualification = receipt.qualification.as_str();
        self.route.wgpu_qualification_provenance = receipt.qualification_provenance.clone();
        self.route.wgpu_failed_rung = receipt.failed_rung.clone();
        self.route.backend_fallback_reason = receipt.fallback_reason.clone();
        self.route.proof_backend = receipt.effective.to_string();
        self.route.proof_backend_provenance = receipt.provenance.clone();
        self
    }

    pub(super) fn terminal_peel_activation(&self) -> AppliedTerminalPeel {
        self.softmax.terminal_peel_activation
    }

    #[allow(clippy::too_many_arguments)]
    fn from_resolved_with_gates(
        config: &BetaCrownConfig,
        is_graph_model: bool,
        late_sequential_conjunction_graph_upgrade: bool,
        use_relu_split: bool,
        gpu_bab: bool,
        run_upfront_pgd: bool,
        vgg_abcrown_treatment_active: bool,
        complete_verifier: CompleteVerifierArg,
        selected_backend: BackendArg,
        proof_backend: String,
        property_is_disjunction: Option<bool>,
        gates: RuntimeTreatmentGates,
    ) -> Self {
        let kfsb_heuristic = matches!(
            config.branching_heuristic,
            BranchingHeuristic::Kfsb | BranchingHeuristic::KfsbInterceptOnly
        );
        let depth_two = config.depth_two_branch_lookahead;
        let depth_two_lookahead_round_zero_supported = depth_two.enabled_at_round(0)
            && matches!(config.branching_heuristic, BranchingHeuristic::Kfsb)
            && !config.verify_upper_bound
            && !gates.depth_two_lookahead_legacy_observer_conflict
            && gates.wave_reduce_op == KfsbReduceOp::Min
            && gates.wave_candidates > 0;
        let typed_depth_two_select = depth_two_lookahead_round_zero_supported
            && depth_two.mode == DepthTwoBranchLookaheadMode::Select;
        let wave_kfsb_armed = gates
            .kfsb_multi_env_override
            .unwrap_or(config.use_kfsb_multi_branching || typed_depth_two_select)
            && kfsb_heuristic
            && config.fsb_candidates > 0;
        let top_level_output_constraint_matrix = config
            .alpha_config
            .output_constraints
            .as_ref()
            .map(|constraints| EffectiveOutputConstraintMatrixProjection {
                rows: constraints.a_matrix.nrows(),
                columns: constraints.a_matrix.ncols(),
                rhs_entries: constraints.rhs.len(),
                is_conjunction: constraints.is_conjunction,
                clause_indices_present: constraints.clause_indices.is_some(),
                clause_count: constraints.clause_indices.as_ref().map(Vec::len),
                clause_row_counts: constraints
                    .clause_indices
                    .as_ref()
                    .map(|clauses| clauses.iter().map(Vec::len).collect()),
            });
        let serial_clause_rebinding = match property_is_disjunction {
            Some(true) => "possible_but_unobserved_for_top_level_disjunctions",
            Some(false) => "not_applicable_for_top_level_conjunction",
            None => "property_shape_unavailable",
        };

        Self {
            schema: EFFECTIVE_TREATMENT_SCHEMA,
            batch: EffectiveBatchProjection {
                configured_size: config.batch_size,
                build_batch_size: config.build_batch_size,
                auto_enlarge: config.auto_enlarge_batch_size,
                adaptive_microbatch_controller_armed: config.auto_enlarge_batch_size
                    && gates.adaptive_microbatch_controller,
                max_relu_split_depth: config.max_relu_split_depth,
                min_fill_ratio: config.min_batch_fill_ratio,
                parallel_children: config.parallel_children,
            },
            branching: EffectiveBranchingProjection {
                heuristic: branching_heuristic_name(&config.branching_heuristic),
                input_split_coeff_threshold: config.input_split_coeff_thresh,
                input_split_adv_check: config.adv_check,
                reorder_bab: config.reorder_bab,
                configured_candidates: config.fsb_candidates,
                effective_wave_candidates: gates.wave_candidates,
                configured_reduce_op: kfsb_reduce_op_name(config.kfsb_reduce_op),
                effective_wave_reduce_op: kfsb_reduce_op_name(gates.wave_reduce_op),
                kfsb_multi_configured: config.use_kfsb_multi_branching,
                kfsb_cert_reuse_configured: config.kfsb_cert_reuse,
                kfsb_cert_reuse_armed: gates.kfsb_cert_reuse_armed,
                multi_objective_critical_kfsb_configured: config.use_multi_objective_critical_kfsb,
                kfsb_multi_env_override: gates.kfsb_multi_env_override,
                depth_two_lookahead_mode: depth_two_branch_lookahead_mode_name(depth_two.mode),
                depth_two_lookahead_candidates: depth_two.candidates,
                depth_two_lookahead_top_rounds: depth_two.top_rounds,
                depth_two_lookahead_discount: depth_two.discount,
                depth_two_lookahead_round_zero_supported,
                adaptive_depth_shadow_env_armed: gates.adaptive_depth_shadow_env_armed,
                adaptive_depth_select_env_armed: gates.adaptive_depth_select_env_armed,
                adaptive_depth_commit_env_armed: gates.adaptive_depth_commit_env_armed,
                depth_two_lookahead_legacy_observer_conflict: gates
                    .depth_two_lookahead_legacy_observer_conflict,
                wave_kfsb_armed,
                scorer_fix_env_armed: gates.scorer_fix,
                competing_branch_experiment_armed: gates.competing_branch_experiment,
            },
            attack: EffectiveAttackProjection {
                enabled: config.enable_pgd_attack,
                schedule: attack_schedule_name(config.enable_pgd_attack, run_upfront_pgd),
                pgd_restarts: config.pgd_restarts,
                pgd_steps: config.pgd_steps,
            },
            alpha_crown: EffectiveAlphaProjection {
                enabled: config.use_alpha_crown,
                iterations: config.alpha_config.iterations,
                learning_rate: config.alpha_config.learning_rate,
                lr_decay: config.alpha_config.lr_decay,
                optimizer: optimizer_name(config.alpha_config.optimizer),
                gradient_method: gradient_method_name(config.alpha_config.gradient_method),
                fix_interm_bounds: config.alpha_config.fix_interm_bounds,
                cgan_sparse_target_complete_root: config
                    .alpha_config
                    .cgan_sparse_target_complete_root,
                cgan_complete_crown_ibp_root: config.alpha_config.cgan_complete_crown_ibp_root,
                full_conv_alpha: config.alpha_config.full_conv_alpha,
                adaptive_skip: config.alpha_config.adaptive_skip,
                adaptive_skip_depth_threshold: config.alpha_config.adaptive_skip_depth_threshold,
                early_stop_patience: config.alpha_config.early_stop_patience,
                start_save_best: config.alpha_config.start_save_best,
                final_bound_only_env_armed: gates.final_alpha_bound_only,
                packed_graph_alpha_queue_env_armed: gates.packed_graph_alpha_queue,
            },
            beta_crown: EffectiveBetaProjection {
                iterations: config.beta_iterations,
                learning_rate_alpha: config.alpha_lr,
                learning_rate_beta: config.beta_lr,
                max_depth: config.beta_max_depth,
                early_stop_patience: config.early_stop_patience,
            },
            clip: EffectiveClipProjection {
                interm_domain: config.enable_clip_interm_domain,
                interm_topk: config.clip_interm_topk,
                in_alpha_crown: config.clip_in_alpha_crown,
                prune: config.clip_interm_prune,
                use_final_layer: config.clip_interm_use_final_layer,
                input_split_fresh_domain_clip_configured: config.input_split_fresh_domain_clip,
            },
            root: EffectiveRootProjection {
                dense_head_configured: config.root_crown_interm_dense_head,
                sparse_configured: config.root_sparse_interm_crown,
                sparse_gate: gates.root_sparse_gate,
                sparse_effective_armed: gates.root_sparse_effective_armed,
                atomic_root_c_margin_iterations: config.atomic_root_c_margin_iterations,
                root_spec_prune_requested: gates.root_spec_prune_requested,
            },
            invprop: EffectiveInvpropProjection {
                enabled: config.alpha_config.invprop.enabled,
                apply_output_constraints_to: config
                    .alpha_config
                    .invprop
                    .apply_output_constraints_to
                    .clone(),
                tighten_input_bounds: config.alpha_config.invprop.tighten_input_bounds,
                best_of_oc_and_no_oc: config.alpha_config.invprop.best_of_oc_and_no_oc,
                directly_optimize: config.alpha_config.invprop.directly_optimize.clone(),
                share_gammas: config.alpha_config.invprop.share_gammas,
                per_layer_gammas: config.alpha_config.invprop.per_layer_gammas,
                optimize_gammas: config.alpha_config.invprop.optimize_gammas,
                gamma_lr: config.alpha_config.invprop.gamma_lr,
                top_level_output_constraint_matrix,
                serial_clause_rebinding,
                split_lift_requested: gates.invprop_split_lift_requested,
                // The env-gated research module has no production call sites.
                // Record requests without representing them as an armed treatment.
                split_lift_effective_armed: false,
            },
            softmax: EffectiveSoftmaxProjection {
                objective_envelope_env_armed: gates.softmax_objective_envelope,
                terminal_peel_requested: false,
                terminal_peel_applied: false,
                terminal_peel_activation: AppliedTerminalPeel::None,
            },
            route: EffectiveRouteProjection {
                model_kind: if is_graph_model {
                    "graph"
                } else {
                    "sequential"
                },
                configured_conv_mode: conv_mode_name(config.conv_mode),
                vgg_abcrown_treatment_active,
                late_sequential_conjunction_graph_upgrade,
                use_relu_split,
                gpu_bab,
                run_upfront_pgd,
                complete_verifier: complete_verifier_name(complete_verifier),
                selected_backend: selected_backend.to_string(),
                requested_backend: selected_backend.to_string(),
                backend_request_source: "unrecorded_internal",
                backend_selection_reason: None,
                effective_backend: proof_backend.clone(),
                wgpu_qualification: "unrecorded",
                wgpu_qualification_provenance: None,
                wgpu_failed_rung: None,
                backend_fallback_reason: None,
                proof_backend,
                proof_backend_provenance: "unrecorded".to_string(),
                property_is_disjunction,
                intermediate_bound_transfer: config.enable_interm_transfer,
                use_forward_bounds: config.use_forward_bounds,
                use_crown_ibp: config.use_crown_ibp,
                input_split_ibp_enhancement: config.input_split_ibp_enhancement,
                input_split_conic_objective_configured: config.input_split_conic_objective,
                input_split_conic_objective_route_eligible: is_graph_model
                    && !use_relu_split
                    && property_is_disjunction == Some(false)
                    && config.input_split_conic_objective_eligible(),
            },
        }
    }
}

fn conv_mode_name(mode: ConvMode) -> &'static str {
    match mode {
        ConvMode::Auto => "auto",
        ConvMode::Patches => "patches",
        ConvMode::Matrix => "matrix",
    }
}

fn attack_schedule_name(enabled: bool, run_upfront_pgd: bool) -> &'static str {
    match (enabled, run_upfront_pgd) {
        (false, _) => "disabled",
        (true, true) => "upfront",
        (true, false) => "input_bab",
    }
}

fn branching_heuristic_name(heuristic: &BranchingHeuristic) -> &'static str {
    match heuristic {
        BranchingHeuristic::LargestBoundWidth => "largest_bound_width",
        BranchingHeuristic::BoundImpact => "bound_impact",
        BranchingHeuristic::FilteredSmartBranching => "filtered_smart_branching",
        BranchingHeuristic::Kfsb => "kfsb",
        BranchingHeuristic::KfsbInterceptOnly => "kfsb_intercept_only",
        BranchingHeuristic::Sequential => "sequential",
        BranchingHeuristic::InputSplit => "input_split",
        BranchingHeuristic::GenBaB(_) => "genbab",
    }
}

fn kfsb_reduce_op_name(op: KfsbReduceOp) -> &'static str {
    match op {
        KfsbReduceOp::Min => "min",
        KfsbReduceOp::Max => "max",
        KfsbReduceOp::Mean => "mean",
    }
}

fn depth_two_branch_lookahead_mode_name(mode: DepthTwoBranchLookaheadMode) -> &'static str {
    match mode {
        DepthTwoBranchLookaheadMode::Off => "off",
        DepthTwoBranchLookaheadMode::Shadow => "shadow",
        DepthTwoBranchLookaheadMode::Select => "select",
    }
}

fn optimizer_name(optimizer: Optimizer) -> &'static str {
    match optimizer {
        Optimizer::Sgd => "sgd",
        Optimizer::Adam => "adam",
    }
}

fn gradient_method_name(method: GradientMethod) -> &'static str {
    match method {
        GradientMethod::FiniteDifferences => "finite_differences",
        GradientMethod::Spsa => "spsa",
        GradientMethod::Analytic => "analytic",
        GradientMethod::AnalyticChain => "analytic_chain",
    }
}

fn complete_verifier_name(verifier: CompleteVerifierArg) -> &'static str {
    match verifier {
        CompleteVerifierArg::Auto => "auto",
        CompleteVerifierArg::Bab => "bab",
        CompleteVerifierArg::Mip => "mip",
    }
}

#[derive(Debug)]
struct CompetitionJsonPayload {
    status: &'static str,
    reason: Option<serde_json::Value>,
    counterexample: Option<(Vec<f32>, Vec<f32>)>,
    property_file: Option<String>,
    epsilon: Option<f32>,
    threshold: f32,
    domains_explored: usize,
    domains_verified: usize,
    cuts_generated: usize,
    max_depth_reached: usize,
    time_elapsed_s: f64,
    output_bound_width: Option<f32>,
    method: Option<String>,
    effective_config: Option<EffectiveTreatmentProjection>,
}

fn json_f32_array(values: &[f32]) -> Vec<serde_json::Value> {
    values.iter().map(|&value| json_f32(value)).collect()
}

/// JSON diagnostic view of X must match the decimal emitted in
/// `counterexample_vnnlib`. Serializing an f32 directly through serde_json can
/// expose its exact f64 promotion instead (0.1f32 =>
/// 0.10000000149011612), creating two contradictory witnesses in one payload.
fn json_emitted_input_array(values: &[f32]) -> Vec<serde_json::Value> {
    values
        .iter()
        .map(|&value| {
            if !value.is_finite() {
                return json_f32(value);
            }
            let parsed = value
                .to_string()
                .parse::<f64>()
                .expect("finite f32 Display must parse as finite f64");
            serde_json::json!(parsed)
        })
        .collect()
}

/// Build a VNN-COMP standard SMT-LIB counterexample witness string.
///
/// The official VNN-COMP counterexample checker expects SMT-LIB-style variable
/// assignments rather than a raw JSON array, e.g.:
/// ```text
/// ((X_0 0.123456)
/// (X_1 -0.654321)
/// (Y_0 1.200000)
/// (Y_1 -0.300000))
/// ```
/// where `X_i` are the flattened network INPUT values (input-tensor flatten
/// order) and `Y_j` the corresponding network OUTPUT values.
///
/// Input formatting deliberately remains the shortest decimal that round-trips
/// to the original `f32`: the trusted gate and the organizer must replay the
/// same input tensor bits.
///
/// Output formatting uses the shortest decimal that round-trips to the exact
/// `f64` promotion of each `f32`.  The VNN-COMP 2025 zero-tolerance checker
/// promotes ONNX Runtime's `f32` output to `f64` before comparing it with the
/// parsed textual `Y`; formatting `Y` as an `f32` (for example `0.1`) therefore
/// created a nonzero comparison error even when the bits came directly from
/// ONNX Runtime.  The exact promotion (for example `0.10000000149011612`) makes
/// that comparison bit-exact without changing the output value.  The 2026
/// checker ignores textual `Y` and replays ONNX Runtime, so this remains
/// forward-compatible; exact input membership is still enforced independently.
fn counterexample_vnnlib(input: &[f32], output: &[f32]) -> String {
    let mut lines = Vec::with_capacity(input.len() + output.len());
    for (i, &value) in input.iter().enumerate() {
        lines.push(format!("(X_{i} {value})"));
    }
    for (j, &value) in output.iter().enumerate() {
        lines.push(format!("(Y_{j} {})", f64::from(value)));
    }
    format!("({})", lines.join("\n"))
}

/// Reconstruct the exact f64 values the organizer parses from our emitted X
/// tokens. `f64::from(value)` is NOT equivalent: `0.1f32` is emitted as `0.1`,
/// while its exact f64 promotion is `0.10000000149011612`.
fn organizer_emitted_input_view(input: &[f32]) -> Option<Vec<f64>> {
    input
        .iter()
        .map(|value| {
            if !value.is_finite() {
                return None;
            }
            value
                .to_string()
                .parse::<f64>()
                .ok()
                .filter(|parsed| parsed.is_finite())
        })
        .collect()
}

/// Validate a public standalone witness against the full property exactly as
/// the emitted SMT-LIB decimals are interpreted.
///
/// This is intentionally a PUBLICATION gate, not an attack-candidate gate.
/// The in-process VNN-COMP caller may need the raw outward-rounded f32 point as
/// a refinement seed (notably when cGAN's pinned f64 decimal is not preserved
/// by f32 textual rendering); its own trusted ORT + exact-f64 gate decides
/// whether a repaired witness may be emitted.
/// Re-execute the witness through ONNX Runtime and confirm the claimed `Y`.
///
/// THE GAP THIS CLOSES. Until this existed the standalone `beta-crown --json`
/// surface published a `Y` produced by NY's OWN forward and never checked
/// against onnxruntime. The organizer does the opposite: it runs the model
/// itself and REJECTS the witness when
/// `||ort_out - Y_file||_inf / ||Y_file||_inf > COUNTEREXAMPLE_RTOL` (1e-3,
/// `SCORING-ZERO-TOL/counterexamples.py` + `settings.py:68`). A `Y` NY never
/// verified is the ONE direction in which NY is LOOSER than the organizer, and
/// it is the direction that scores `PENALTY_INCORRECT` (-150) rather than 0:
/// `process_results.py` charges the penalty for `res == "violated"` with no
/// valid CE for that tool.
///
/// Mirrors `vnncomp::confirm_violation_with_ort` exactly: the organizer parses
/// the witness decimals AS WRITTEN (f64) for membership and runs inference on
/// the f32 tensor, so the f64 view drives the property decision while the f32
/// cast drives the forward.
///
/// Returns `None` when the check could not be RUN (no model path, or the
/// session/inference failed). The caller treats that as "unconfirmed" and keeps
/// the pre-existing conservative behaviour — it must never read as confirmed.
/// Outcome of re-executing the witness through ONNX Runtime.
///
/// The three states must stay distinct. Collapsing "could not run" into
/// "mismatched" turns every environment without a usable ORT session into a
/// blanket refusal of valid witnesses — a self-inflicted score loss, and the
/// exact bug the first draft of this gate shipped.
enum OrtConfirmation {
    /// ORT ran and its output agrees with the claimed `Y` within the
    /// organizer's exec-match tolerance. Carries the ORT output, which is what
    /// the spec check should see.
    Confirmed(Vec<f32>),
    /// ORT ran and DISAGREED beyond tolerance. The organizer would reject this
    /// witness, so publishing it risks the -150 rather than a forgone +10.
    Mismatched,
    /// The check could not be performed (no model path, or the session or
    /// inference failed). Says NOTHING about the witness.
    Unavailable,
}

/// Re-execute the witness through ONNX Runtime and confirm the claimed `Y`.
///
/// THE GAP THIS CLOSES. The standalone `beta-crown --json` surface published a
/// `Y` produced by NY's OWN forward and never checked against onnxruntime. The
/// organizer does the opposite: it runs the model itself and REJECTS the witness
/// when `||ort_out - Y_file||_inf / ||Y_file||_inf > COUNTEREXAMPLE_RTOL` (1e-3;
/// `SCORING-ZERO-TOL/counterexamples.py`, `settings.py:68`). An unverified `Y`
/// is the one direction in which NY is LOOSER than the organizer, and it is the
/// direction that scores `PENALTY_INCORRECT` (-150) instead of 0.
///
/// Mirrors `vnncomp::confirm_violation_with_ort`: the organizer parses the
/// witness decimals AS WRITTEN (f64) for membership and runs inference on the
/// f32 tensor, so the f32 cast drives the forward here while the f64 view drives
/// the property decision downstream.
fn ort_confirmed_output(model: Option<&Path>, input: &[f32], claimed: &[f32]) -> OrtConfirmation {
    let Some(model) = model else {
        return OrtConfirmation::Unavailable;
    };
    let Ok(mut forward) = ny_onnx::diff::OrtForward::from_path(model, input.len()) else {
        return OrtConfirmation::Unavailable;
    };
    let Ok(produced) = forward.run(input) else {
        return OrtConfirmation::Unavailable;
    };
    if produced.len() != claimed.len() || !produced.iter().all(|v| v.is_finite()) {
        // A length or finiteness disagreement IS a real disagreement, not an
        // environment problem.
        return OrtConfirmation::Mismatched;
    }
    // The organizer's exec-match gate, same shape and same constant.
    const COUNTEREXAMPLE_RTOL: f32 = 1e-3;
    let scale = claimed.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    if scale >= 1e-6 {
        let diff = produced
            .iter()
            .zip(claimed)
            .fold(0.0f32, |a, (&p, &c)| a.max((p - c).abs()));
        if diff / scale > COUNTEREXAMPLE_RTOL {
            return OrtConfirmation::Mismatched;
        }
    }
    OrtConfirmation::Confirmed(produced)
}

fn standalone_witness_matches_spec(spec: &VnnLibSpec, input: &[f32], output: &[f32]) -> bool {
    if spec.dual_network.is_some()
        || input.len() != spec.num_inputs
        || output.len() != spec.num_outputs
        || !output.iter().all(|value| value.is_finite())
    {
        return false;
    }
    let Some(input64) = organizer_emitted_input_view(input) else {
        return false;
    };
    let output64: Vec<f64> = output.iter().map(|&value| f64::from(value)).collect();
    crate::commands::vnncomp::property_violated_f64(spec, &input64, &output64)
}

/// Apply the organizer-exact property check shared by every standalone
/// publication surface. The caller must supply output values in original-model
/// coordinates; a terminal peel that cannot provide those values is refused
/// before this helper is called.
fn standalone_witness_matches_property(
    property: Option<&Path>,
    input: &[f32],
    output: &[f32],
) -> bool {
    if let Some(property) = property {
        match ny_onnx::vnnlib::load_vnnlib(property) {
            Ok(spec) => standalone_witness_matches_spec(&spec, input, output),
            Err(error) => {
                tracing::warn!(
                    property = %property.display(),
                    %error,
                    "Unable to reload VNN-LIB for standalone witness publication; refusing Violated"
                );
                false
            }
        }
    } else {
        // Propertyless threshold mode has no organizer VNN-LIB box to replay,
        // but no public surface may carry a non-finite assignment.
        input.iter().all(|value| value.is_finite()) && output.iter().all(|value| value.is_finite())
    }
}

/// Check the typed BaB witness exactly as the standalone renderer would expose
/// it. This is deliberately separate from capture: the in-process VNN-COMP
/// caller retains raw candidates for its stronger ORT/refinement gate.
fn standalone_bab_witness_is_publishable(
    result: &BetaCrownResult,
    property: Option<&Path>,
    applied_terminal_peel: AppliedTerminalPeel,
) -> bool {
    let BabVerificationStatus::Violated {
        counterexample,
        output,
    } = &result.result
    else {
        return true;
    };
    let Some(original_output) = applied_terminal_peel.output_in_original_coordinates(output) else {
        return false;
    };
    standalone_witness_matches_property(property, counterexample, &original_output)
}

/// Seal the standalone `beta-crown --json` witness surface. Attack and bound
/// boxes use outward-rounded f32 endpoints for verification soundness; a point
/// on that widened shell is a useful internal candidate but is not a valid
/// organizer witness. Captured in-process VNN-COMP results deliberately bypass
/// this seam and are handled by their stronger trusted-ORT/refinement gate.
fn seal_standalone_witness(
    payload: &mut CompetitionJsonPayload,
    property: Option<&Path>,
    model: Option<&Path>,
) -> bool {
    let Some((input, output)) = payload.counterexample.as_ref() else {
        return false;
    };

    // ORGANIZER ORDER: the exec-match gate runs BEFORE the spec gate. Confirm the
    // claimed `Y` against onnxruntime first, and let the spec check see the ORT
    // output rather than NY's own forward — that is the pair the organizer
    // actually scores. `None` means the check could not RUN (no model path or a
    // session/inference failure), never that it passed, so a reachable model
    // that fails to confirm REFUSES publication.
    let confirmation = ort_confirmed_output(model, input, output);
    if matches!(confirmation, OrtConfirmation::Mismatched) {
        tracing::warn!(
            "Standalone counterexample did not survive ONNX Runtime re-execution; \
             refusing Violated rather than publishing a Y the organizer would reject"
        );
        payload.status = "unknown";
        payload.reason = Some(serde_json::json!(
            "counterexample failed ONNX Runtime re-execution"
        ));
        payload.counterexample = None;
        return true;
    }
    // `Unavailable` falls through to the pre-existing organizer-exact spec check
    // on NY's own output: strictly no worse than before this gate existed.
    let checked_output: &[f32] = match &confirmation {
        OrtConfirmation::Confirmed(ort) => ort,
        _ => output,
    };

    let valid = standalone_witness_matches_property(property, input, checked_output);

    if !valid {
        tracing::warn!(
            "Standalone counterexample failed organizer-exact publication validation; \
             downgrading Violated to Unknown"
        );
        payload.status = "unknown";
        payload.reason = Some(serde_json::json!(
            "counterexample failed organizer-exact publication validation"
        ));
        payload.counterexample = None;
        return true;
    }
    false
}

fn bounded_tensor_width(result: &BetaCrownResult) -> Option<f32> {
    result.output_bounds.as_ref().and_then(|bounds| {
        let width = bounds.width().iter().copied().fold(0.0f32, f32::max);
        width.is_finite().then_some(width)
    })
}

#[cfg(feature = "mip")]
fn bounds_width(bounds: &[Bound]) -> Option<f32> {
    let width = bounds
        .iter()
        .map(|bound| bound.upper() - bound.lower())
        .fold(0.0f32, f32::max);
    width.is_finite().then_some(width)
}

fn format_competition_json(payload: CompetitionJsonPayload) -> Result<String> {
    let execution_observations = ny_propagate::execution_telemetry::snapshot();
    format_competition_json_with_observations(payload, &execution_observations)
}

fn format_competition_json_with_observations(
    payload: CompetitionJsonPayload,
    execution_observations: &ny_propagate::execution_telemetry::ExecutionObservations,
) -> Result<String> {
    use serde_json::json;

    let counterexample_vnnlib = payload
        .counterexample
        .as_ref()
        .map(|(input, output)| counterexample_vnnlib(input, output));

    let mut json_output = json!({
        "status": payload.status,
        "reason": payload.reason,
        "counterexample": payload.counterexample.as_ref().map(|(input, output)| {
            json!({
                "input": json_emitted_input_array(input),
                "output": json_f32_array(output),
            })
        }),
        // VNN-COMP standard SMT-LIB witness (single string, newline-separated
        // X_i/Y_j assignments). Consumed by run_instance.sh for the witness file.
        "counterexample_vnnlib": counterexample_vnnlib,
        "property_file": payload.property_file,
        "epsilon": payload.epsilon,
        "threshold": payload.threshold,
        "domains_explored": payload.domains_explored,
        "domains_verified": payload.domains_verified,
        "cuts_generated": payload.cuts_generated,
        "max_depth_reached": payload.max_depth_reached,
        "time_elapsed_s": payload.time_elapsed_s,
        "output_bound_width": payload.output_bound_width,
        // Observed run state is deliberately a sibling of effective_config:
        // counters vary by instance and must never perturb treatment hashes.
        "execution_observations": execution_observations,
    });

    if let Some(method) = payload.method {
        json_output["method"] = json!(method);
    }
    if let Some(effective_config) = payload.effective_config {
        json_output["effective_config"] = serde_json::to_value(effective_config)?;
    }

    Ok(serde_json::to_string_pretty(&json_output)?)
}

fn output_in_original_coordinates(
    output: &[f32],
    applied_terminal_peel: AppliedTerminalPeel,
) -> Vec<f32> {
    // Softmax-family values stay in peeled coordinates for in-process capture;
    // the outer VNN-COMP seam rehydrates them with the original ONNX model.
    // Standalone publication is refused below before those values can escape.
    applied_terminal_peel
        .output_in_original_coordinates(output)
        .map(std::borrow::Cow::into_owned)
        .unwrap_or_else(|| output.to_vec())
}

/// A Softmax-family peel leaves a concrete witness in logit coordinates.
/// Those logits preserve the relational property but are not an admissible
/// original-model `Y` assignment.  In-process VNN-COMP capture intentionally
/// keeps the seed so its outer ORT seam can rehydrate it; standalone JSON has
/// no original-model forward here and must refuse the witness.
fn refuse_unrehydrated_softmax_family_witness(
    payload: &mut CompetitionJsonPayload,
    applied_terminal_peel: AppliedTerminalPeel,
) -> bool {
    if !applied_terminal_peel.needs_original_output_rehydration()
        || payload.counterexample.is_none()
    {
        return false;
    }
    tracing::warn!(
        ?applied_terminal_peel,
        "Standalone peeled Softmax-family counterexample has logit-space outputs; \
         downgrading Violated to Unknown until original outputs are rehydrated"
    );
    payload.status = "unknown";
    payload.reason = Some(serde_json::json!(
        "peeled Softmax-family counterexample requires original-model output rehydration"
    ));
    payload.counterexample = None;
    true
}

fn beta_crown_json_payload(
    result: &BetaCrownResult,
    property: Option<&Path>,
    epsilon: f32,
    effective_threshold: f32,
    applied_terminal_peel: AppliedTerminalPeel,
    effective_config: Option<&EffectiveTreatmentProjection>,
) -> CompetitionJsonPayload {
    let (status, reason, counterexample) = match &result.result {
        BabVerificationStatus::Verified => ("verified", None, None),
        BabVerificationStatus::Violated {
            counterexample,
            output,
        } => (
            "violated",
            None,
            Some((
                counterexample.clone(),
                output_in_original_coordinates(output, applied_terminal_peel),
            )),
        ),
        BabVerificationStatus::PotentialViolation { .. } => ("potential_violation", None, None),
        BabVerificationStatus::Unknown { reason } => {
            ("unknown", Some(serde_json::json!(reason)), None)
        }
        BabVerificationStatus::Timeout => ("timeout", None, None),
    };

    CompetitionJsonPayload {
        status,
        reason,
        counterexample,
        property_file: property.map(|path| path.display().to_string()),
        epsilon: if property.is_none() {
            Some(epsilon)
        } else {
            None
        },
        threshold: effective_threshold,
        domains_explored: result.domains_explored,
        domains_verified: result.domains_verified,
        cuts_generated: result.cuts_generated,
        max_depth_reached: result.max_depth_reached,
        time_elapsed_s: result.time_elapsed.as_secs_f64(),
        output_bound_width: bounded_tensor_width(result),
        method: None,
        effective_config: effective_config.cloned(),
    }
}

#[cfg(feature = "mip")]
fn verification_result_json_payload(
    result: &VerificationResult,
    property: Option<&Path>,
    epsilon: f32,
    threshold: f32,
    elapsed: Duration,
    method: &str,
    effective_config: Option<&EffectiveTreatmentProjection>,
) -> Result<CompetitionJsonPayload> {
    let applied_terminal_peel = effective_config
        .map(|config| config.softmax.terminal_peel_activation)
        .unwrap_or_default();
    let (status, reason, counterexample, output_bound_width) = match result {
        VerificationResult::Verified { output_bounds, .. } => {
            ("verified", None, None, bounds_width(output_bounds))
        }
        VerificationResult::Violated {
            counterexample,
            output,
            ..
        } => (
            "violated",
            None,
            Some((
                counterexample.clone(),
                output_in_original_coordinates(output, applied_terminal_peel),
            )),
            None,
        ),
        VerificationResult::Unknown { reason, bounds, .. } => (
            "unknown",
            Some(serde_json::to_value(reason)?),
            None,
            bounds_width(bounds),
        ),
        VerificationResult::Timeout { partial_bounds, .. } => (
            "timeout",
            None,
            None,
            partial_bounds.as_deref().and_then(bounds_width),
        ),
    };

    Ok(CompetitionJsonPayload {
        status,
        reason,
        counterexample,
        property_file: property.map(|path| path.display().to_string()),
        epsilon: if property.is_none() {
            Some(epsilon)
        } else {
            None
        },
        threshold,
        domains_explored: 0,
        domains_verified: 0,
        cuts_generated: 0,
        max_depth_reached: 0,
        time_elapsed_s: elapsed.as_secs_f64(),
        output_bound_width,
        method: Some(method.to_string()),
        effective_config: effective_config.cloned(),
    })
}

/// Format an MIP verdict for an external JSON consumer and return whether a
/// purported Violated witness was refused by the organizer-exact publication
/// gate. Callers must map a refusal to the Unknown exit code.
#[cfg(feature = "mip")]
pub(super) fn format_verification_result_json_for_publication(
    result: &VerificationResult,
    property: Option<&Path>,
    model: Option<&Path>,
    epsilon: f32,
    threshold: f32,
    elapsed: Duration,
    method: &str,
    effective_config: Option<&EffectiveTreatmentProjection>,
) -> Result<(String, bool)> {
    let applied_terminal_peel = effective_config
        .map(|config| config.softmax.terminal_peel_activation)
        .unwrap_or_default();
    let mut payload = verification_result_json_payload(
        result,
        property,
        epsilon,
        threshold,
        elapsed,
        method,
        effective_config,
    )?;
    let publication_refused = if is_capturing() {
        false
    } else {
        refuse_unrehydrated_softmax_family_witness(&mut payload, applied_terminal_peel)
            || seal_standalone_witness(&mut payload, property, model)
    };
    Ok((format_competition_json(payload)?, publication_refused))
}

/// Test-only face that drops the refusal flag. It passes `None` for the model
/// deliberately: with no ONNX file to re-execute through, the standalone-witness
/// seal reports `Unavailable` rather than confirmed or mismatched, which is the
/// state these payload-shape tests want to hold fixed.
#[cfg(all(feature = "mip", test))]
pub(super) fn format_verification_result_json(
    result: &VerificationResult,
    property: Option<&Path>,
    epsilon: f32,
    threshold: f32,
    elapsed: Duration,
    method: &str,
    effective_config: Option<&EffectiveTreatmentProjection>,
) -> Result<String> {
    format_verification_result_json_for_publication(
        result,
        property,
        None,
        epsilon,
        threshold,
        elapsed,
        method,
        effective_config,
    )
    .map(|(rendered, _)| rendered)
}

#[cfg(feature = "mip")]
pub(super) fn verification_result_exit_code(result: &VerificationResult) -> i32 {
    match result {
        VerificationResult::Verified { .. } => exit_codes::VERIFIED,
        VerificationResult::Violated { .. } => exit_codes::VIOLATED,
        VerificationResult::Unknown { .. } => exit_codes::UNKNOWN,
        VerificationResult::Timeout { .. } => exit_codes::TIMEOUT,
    }
}

fn beta_crown_exit_code(status: &BabVerificationStatus) -> i32 {
    match status {
        BabVerificationStatus::Verified => exit_codes::VERIFIED,
        BabVerificationStatus::Violated { .. } => exit_codes::VIOLATED,
        // #3678 rewrites property-backed PotentialViolation to Violated/Unknown before
        // this renderer. The remaining surface is the propertyless threshold path, so
        // shell-level status should stay conservative rather than imply a confirmed SAT.
        BabVerificationStatus::PotentialViolation { .. }
        | BabVerificationStatus::Unknown { .. } => exit_codes::UNKNOWN,
        BabVerificationStatus::Timeout => exit_codes::TIMEOUT,
    }
}

/// Output verification result.
///
/// `verify_upper` selects the direction words in the human-readable report:
/// upper-bound specs (`Y_i >= c` unsafe) prove `output < c`, lower-bound specs
/// prove `output > c`. The JSON payload carries only the numeric threshold, so
/// it does not depend on the flag.
pub(super) fn output_result(
    result: &BetaCrownResult,
    property: &Option<PathBuf>,
    model: Option<&Path>,
    epsilon: f32,
    effective_threshold: f32,
    verify_upper: bool,
    json: bool,
    sigmoid_peeled: bool,
    effective_config: &EffectiveTreatmentProjection,
) -> Result<()> {
    let mut publication_refused = false;
    let applied_terminal_peel = effective_config.terminal_peel_activation();
    debug_assert_eq!(sigmoid_peeled, applied_terminal_peel.is_sigmoid());
    if json {
        let mut payload = beta_crown_json_payload(
            result,
            property.as_deref(),
            epsilon,
            effective_threshold,
            applied_terminal_peel,
            Some(effective_config),
        );
        if !is_capturing() {
            publication_refused =
                refuse_unrehydrated_softmax_family_witness(&mut payload, applied_terminal_peel)
                    || seal_standalone_witness(&mut payload, property.as_deref(), model);
        }
        let rendered = format_competition_json(payload)?;
        if emit_competition_json(&rendered) {
            // Verdict captured in-process (vnncomp path): do NOT print or exit;
            // the caller reads it via `take_captured_json` and translates it.
            return Ok(());
        }
    } else {
        if !is_capturing()
            && !standalone_bab_witness_is_publishable(
                result,
                property.as_deref(),
                applied_terminal_peel,
            )
        {
            tracing::warn!(
                "Standalone counterexample failed organizer-exact publication validation; \
                 downgrading Violated to Unknown"
            );
            publication_refused = true;
        }
        // Mirrors the "Threshold: ... (verifying ...)" line printed at dispatch
        // time: a proven upper bound means every output is BELOW the threshold
        // and a counterexample is one at or above it.
        let (proven, violated, potential) = if verify_upper {
            ("<", ">=", ">=")
        } else {
            (">", "<=", "<")
        };
        println!("\n--- Result ---");
        if publication_refused {
            println!("Status: UNKNOWN");
            println!("Reason: counterexample failed organizer-exact publication validation");
        } else {
            match &result.result {
                BabVerificationStatus::Verified => {
                    println!("Status: VERIFIED");
                    if applied_terminal_peel.needs_original_output_rehydration() {
                        println!(
                            "Equivalent peeled {} preactivation property verified \
                         (not an original-model Y value claim)",
                            applied_terminal_peel.activation_name()
                        );
                    } else {
                        println!(
                            "All inputs produce output {} {}",
                            proven, effective_threshold
                        );
                    }
                }
                BabVerificationStatus::Violated {
                    counterexample,
                    output,
                } => {
                    println!("Status: VIOLATED");
                    if applied_terminal_peel.needs_original_output_rehydration() {
                        println!(
                        "Found counterexample to the equivalent peeled {} preactivation property \
                         (not an original-model Y value claim)",
                        applied_terminal_peel.activation_name()
                    );
                    } else {
                        println!(
                            "Found counterexample where output {} {}",
                            violated, effective_threshold
                        );
                    }
                    println!("Counterexample input: {:?}", counterexample);
                    let (label, displayed_output) =
                        applied_terminal_peel.human_witness_output(output);
                    println!("{label}: {:?}", displayed_output);
                }
                BabVerificationStatus::PotentialViolation { .. } => {
                    println!("Status: POTENTIAL VIOLATION");
                    if applied_terminal_peel.needs_original_output_rehydration() {
                        println!(
                        "Found region where the peeled {} preactivation property may be violated \
                         (not an original-model Y value claim)",
                        applied_terminal_peel.activation_name()
                    );
                    } else {
                        println!(
                            "Found region where output may be {} {}",
                            potential, effective_threshold
                        );
                    }
                }
                BabVerificationStatus::Unknown { reason } => {
                    println!("Status: UNKNOWN");
                    println!("Reason: {}", reason);
                }
                BabVerificationStatus::Timeout => {
                    println!("Status: TIMEOUT");
                    println!("Verification timed out before completion");
                }
            }
        }
        println!("Domains explored: {}", result.domains_explored);
        println!("Domains verified: {}", result.domains_verified);
        if result.cuts_generated > 0 {
            println!("Cuts generated: {}", result.cuts_generated);
        }
        println!("Max depth reached: {}", result.max_depth_reached);
        println!("Time elapsed: {:.2}s", result.time_elapsed.as_secs_f64());

        if let Some(bounds) = &result.output_bounds {
            let width = bounds.width();
            let max_width = width.iter().cloned().fold(0.0f32, f32::max);
            println!("Output bound width: {:.6e}", max_width);
        }
    }

    // Apply exit codes matching the verify command contract.
    // Per designs/2026-02-03-smt-result-semantics.md.
    // Keep the process contract aligned with the JSON contract. A refused
    // witness is Unknown, never the original Violated/SAT exit code.
    let exit_code = if publication_refused {
        exit_codes::UNKNOWN
    } else {
        beta_crown_exit_code(&result.result)
    };
    if exit_code != exit_codes::VERIFIED {
        #[cfg(test)]
        return Ok(());
        #[cfg(not(test))]
        std::process::exit(exit_code);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use std::io::Write;
    use std::time::Duration;

    fn treatment_gates(config: &BetaCrownConfig) -> RuntimeTreatmentGates {
        RuntimeTreatmentGates {
            adaptive_microbatch_controller: false,
            kfsb_multi_env_override: None,
            kfsb_cert_reuse_armed: config.kfsb_cert_reuse,
            wave_candidates: config.fsb_candidates,
            wave_reduce_op: config.kfsb_reduce_op,
            scorer_fix: false,
            competing_branch_experiment: false,
            final_alpha_bound_only: false,
            packed_graph_alpha_queue: false,
            root_sparse_gate: "absent",
            root_sparse_effective_armed: config.root_sparse_interm_crown,
            root_spec_prune_requested: false,
            adaptive_depth_shadow_env_armed: false,
            adaptive_depth_select_env_armed: false,
            adaptive_depth_commit_env_armed: false,
            depth_two_lookahead_legacy_observer_conflict: false,
            invprop_split_lift_requested: false,
            softmax_objective_envelope: false,
        }
    }

    fn exact_point_one_tenth_spec() -> VnnLibSpec {
        let mut spec = VnnLibSpec::new();
        spec.num_inputs = 1;
        spec.num_outputs = 1;
        spec.input_bounds = vec![(0.1, 0.1)];
        spec.declared_input_bounds = spec.input_bounds.clone();
        spec.output_constraints = vec![ny_onnx::vnnlib::OutputConstraint::GreaterEqConst(0, 0.0)];
        spec
    }

    fn violated_result(input: f32) -> BetaCrownResult {
        BetaCrownResult {
            result: BabVerificationStatus::Violated {
                counterexample: vec![input],
                output: vec![1.0],
            },
            domains_explored: 0,
            time_elapsed: Duration::ZERO,
            max_depth_reached: 0,
            output_bounds: None,
            cuts_generated: 0,
            domains_verified: 0,
        }
    }

    fn default_treatment() -> EffectiveTreatmentProjection {
        let config = BetaCrownConfig::default();
        EffectiveTreatmentProjection::from_resolved(
            &config,
            true,
            false,
            true,
            false,
            true,
            false,
            CompleteVerifierArg::Auto,
            BackendArg::Cpu,
            "cpu",
            None,
        )
    }

    #[test]
    fn effective_treatment_seals_full_backend_fallback_receipt() {
        use crate::commands::backend::{BackendRequest, BackendRequestSource, ProofBackendReceipt};

        let receipt = ProofBackendReceipt::refused_wgpu(
            BackendRequest {
                backend: BackendArg::Wgpu,
                source: BackendRequestSource::Preset,
                selection_reason: None,
            },
            "compute-device-cpu",
            Some("Apple M5 (IntegratedGpu, Metal)".to_string()),
            Some("gradual_underflow".to_string()),
            "live adapter qualification refused",
        );
        let projection = default_treatment().with_backend_receipt(&receipt);
        let json = serde_json::to_value(projection).expect("projection serializes");
        let route = &json["route"];

        assert_eq!(route["selected_backend"], "wgpu");
        assert_eq!(route["requested_backend"], "wgpu");
        assert_eq!(route["backend_request_source"], "preset");
        assert_eq!(route["effective_backend"], "cpu");
        assert_eq!(route["wgpu_qualification"], "refused");
        assert_eq!(
            route["wgpu_qualification_provenance"],
            "Apple M5 (IntegratedGpu, Metal)"
        );
        assert_eq!(route["wgpu_failed_rung"], "gradual_underflow");
        assert_eq!(
            route["backend_fallback_reason"],
            "live adapter qualification refused"
        );
        assert_eq!(route["proof_backend"], "cpu");
        assert_eq!(route["proof_backend_provenance"], "compute-device-cpu");
    }

    #[test]
    fn effective_treatment_reports_conic_objective_configuration_and_route_eligibility() {
        let mut config = BetaCrownConfig {
            input_split_conic_objective: true,
            ..Default::default()
        };

        let certificate_projection = EffectiveTreatmentProjection::from_resolved(
            &config,
            true,
            false,
            false,
            false,
            false,
            false,
            CompleteVerifierArg::Auto,
            BackendArg::Cpu,
            "cpu",
            Some(false),
        );
        let certificate_json =
            serde_json::to_value(certificate_projection).expect("projection serializes");
        assert_eq!(
            certificate_json["route"]["input_split_conic_objective_configured"],
            true
        );
        assert_eq!(
            certificate_json["route"]["input_split_conic_objective_route_eligible"], false,
            "certificate-export authority cannot admit a synthetic objective"
        );

        config.verification_artifact_authority =
            ny_propagate::VerificationArtifactAuthority::VerdictOnly;
        let eligible_projection = EffectiveTreatmentProjection::from_resolved(
            &config,
            true,
            false,
            false,
            false,
            false,
            false,
            CompleteVerifierArg::Auto,
            BackendArg::Cpu,
            "cpu",
            Some(false),
        );
        let eligible_json =
            serde_json::to_value(eligible_projection).expect("projection serializes");
        assert_eq!(
            eligible_json["route"]["input_split_conic_objective_route_eligible"],
            true
        );

        let relu_projection = EffectiveTreatmentProjection::from_resolved(
            &config,
            true,
            false,
            true,
            false,
            false,
            false,
            CompleteVerifierArg::Auto,
            BackendArg::Cpu,
            "cpu",
            Some(false),
        );
        let relu_json = serde_json::to_value(relu_projection).expect("projection serializes");
        assert_eq!(
            relu_json["route"]["input_split_conic_objective_route_eligible"], false,
            "ReLU splitting must remain on the historical objective set"
        );
    }

    #[test]
    fn effective_treatment_json_distinguishes_tinyimagenet_off_and_upstream_v25_on() {
        let mut off_config = BetaCrownConfig {
            batch_size: 128,
            auto_enlarge_batch_size: true,
            branching_heuristic: BranchingHeuristic::Kfsb,
            fsb_candidates: 10,
            kfsb_reduce_op: KfsbReduceOp::Max,
            use_kfsb_multi_branching: false,
            use_multi_objective_critical_kfsb: true,
            alpha_lr: 0.1,
            beta_lr: 0.15,
            beta_iterations: 15,
            enable_clip_interm_domain: false,
            clip_interm_topk: 3,
            clip_in_alpha_crown: false,
            conv_mode: ConvMode::Patches,
            enable_pgd_attack: true,
            pgd_restarts: 10,
            ..BetaCrownConfig::default()
        };
        off_config.alpha_config.iterations = 20;
        off_config.alpha_config.learning_rate = 0.2;

        let off = EffectiveTreatmentProjection::from_resolved_with_gates(
            &off_config,
            true,
            false,
            true,
            false,
            true,
            false,
            CompleteVerifierArg::Auto,
            BackendArg::Wgpu,
            "cpu".to_string(),
            Some(true),
            treatment_gates(&off_config),
        );

        let mut on_config = off_config;
        on_config.batch_size = 256;
        on_config.fsb_candidates = 7;
        on_config.use_kfsb_multi_branching = true;
        on_config.kfsb_cert_reuse = true;
        on_config.use_multi_objective_critical_kfsb = false;
        on_config.alpha_config.learning_rate = 0.25;
        on_config.alpha_config.cgan_sparse_target_complete_root = true;
        on_config.alpha_config.cgan_complete_crown_ibp_root = true;
        on_config.beta_iterations = 8;
        on_config.enable_clip_interm_domain = true;
        on_config.clip_interm_topk = 20;
        on_config.build_batch_size = Some(512);
        on_config.conv_mode = ConvMode::Matrix;
        on_config.pgd_restarts = 100;
        on_config.input_split_coeff_thresh = 0.01;
        on_config.reorder_bab = true;
        on_config.root_crown_interm_dense_head = true;
        on_config.root_sparse_interm_crown = true;
        on_config.input_split_fresh_domain_clip = true;
        let on = EffectiveTreatmentProjection::from_resolved_with_gates(
            &on_config,
            true,
            false,
            true,
            false,
            true,
            false,
            CompleteVerifierArg::Auto,
            BackendArg::Wgpu,
            "cpu".to_string(),
            Some(true),
            treatment_gates(&on_config),
        );

        let off_json = serde_json::to_value(&off).expect("OFF projection serializes");
        let on_json = serde_json::to_value(&on).expect("ON projection serializes");
        assert_eq!(off_json["schema"], EFFECTIVE_TREATMENT_SCHEMA);
        assert_eq!(on_json["schema"], EFFECTIVE_TREATMENT_SCHEMA);
        assert_eq!(off_json["batch"]["configured_size"], 128);
        assert_eq!(on_json["batch"]["configured_size"], 256);
        assert_eq!(on_json["batch"]["build_batch_size"], 512);
        assert_eq!(off_json["branching"]["heuristic"], "kfsb");
        assert_eq!(on_json["branching"]["heuristic"], "kfsb");
        assert_eq!(off_json["branching"]["configured_candidates"], 10);
        assert_eq!(on_json["branching"]["configured_candidates"], 7);
        assert_eq!(
            on_json["branching"]["input_split_coeff_threshold"],
            serde_json::json!(f64::from(0.01f32))
        );
        assert_eq!(off_json["branching"]["input_split_adv_check"], -1);
        assert_eq!(on_json["branching"]["reorder_bab"], true);
        assert_eq!(off_json["branching"]["kfsb_multi_configured"], false);
        assert_eq!(on_json["branching"]["kfsb_multi_configured"], true);
        assert_eq!(off_json["branching"]["kfsb_cert_reuse_configured"], false);
        assert_eq!(off_json["branching"]["kfsb_cert_reuse_armed"], false);
        assert_eq!(on_json["branching"]["kfsb_cert_reuse_configured"], true);
        assert_eq!(on_json["branching"]["kfsb_cert_reuse_armed"], true);
        assert_eq!(off_json["branching"]["wave_kfsb_armed"], false);
        assert_eq!(on_json["branching"]["wave_kfsb_armed"], true);
        assert_eq!(off_json["branching"]["depth_two_lookahead_mode"], "off");
        assert_eq!(on_json["branching"]["depth_two_lookahead_mode"], "off");
        assert_eq!(
            off_json["branching"]["multi_objective_critical_kfsb_configured"],
            true
        );
        assert_eq!(
            on_json["branching"]["multi_objective_critical_kfsb_configured"],
            false
        );
        assert_eq!(
            off_json["alpha_crown"]["learning_rate"],
            serde_json::json!(f64::from(0.2f32))
        );
        assert_eq!(
            on_json["alpha_crown"]["learning_rate"],
            serde_json::json!(f64::from(0.25f32))
        );
        assert_eq!(
            on_json["alpha_crown"]["cgan_sparse_target_complete_root"],
            true
        );
        assert_eq!(on_json["alpha_crown"]["cgan_complete_crown_ibp_root"], true);
        assert_eq!(off_json["beta_crown"]["iterations"], 15);
        assert_eq!(on_json["beta_crown"]["iterations"], 8);
        assert_eq!(off_json["clip"]["interm_domain"], false);
        assert_eq!(on_json["clip"]["interm_domain"], true);
        assert_eq!(off_json["clip"]["interm_topk"], 3);
        assert_eq!(on_json["clip"]["interm_topk"], 20);
        assert_eq!(on_json["clip"]["in_alpha_crown"], false);
        assert_eq!(
            off_json["clip"]["input_split_fresh_domain_clip_configured"],
            false
        );
        assert_eq!(
            on_json["clip"]["input_split_fresh_domain_clip_configured"],
            true
        );
        assert_eq!(on_json["attack"]["pgd_restarts"], 100);
        assert_eq!(on_json["attack"]["schedule"], "upfront");
        assert_eq!(on_json["root"]["dense_head_configured"], true);
        assert_eq!(on_json["root"]["sparse_configured"], true);
        assert_eq!(on_json["root"]["sparse_effective_armed"], true);
        assert_eq!(on_json["route"]["model_kind"], "graph");
        assert_eq!(on_json["route"]["configured_conv_mode"], "matrix");
        assert_eq!(on_json["route"]["vgg_abcrown_treatment_active"], false);
        assert_eq!(on_json["route"]["selected_backend"], "wgpu");
        assert_eq!(on_json["route"]["proof_backend"], "cpu");

        let payload = beta_crown_json_payload(
            &BetaCrownResult {
                result: BabVerificationStatus::Verified,
                domains_explored: 1,
                time_elapsed: Duration::from_secs(1),
                max_depth_reached: 0,
                output_bounds: None,
                cuts_generated: 0,
                domains_verified: 1,
            },
            None,
            0.02,
            0.0,
            AppliedTerminalPeel::None,
            Some(&on),
        );
        // Pins one committed exact-C outcome: 2 of 4 attempted iterations
        // accepted under a 4-iteration limit, an uncompressed 3-row selection,
        // and authenticated multi-iteration evidence with MW not requested.
        // Every other observation group stays at its `Default`, which is what
        // the fresh-domain-clip assertions below read. `ExactCObservations`
        // carries a private runtime-only field, so outside `ny-propagate` its
        // counters must be assigned rather than written as a struct literal.
        let mut observations = ny_propagate::execution_telemetry::ExecutionObservations {
            run_active: true,
            ..Default::default()
        };
        observations.exact_c.observed = true;
        observations.exact_c.selections = 1;
        observations.exact_c.selected_iteration_limit = Some(4);
        observations.exact_c.selected_compressed = Some(false);
        observations.exact_c.layout_observations = 1;
        observations.exact_c.source_rows = 3;
        observations.exact_c.evaluated_rows = 3;
        observations.exact_c.outcomes_observed = 1;
        observations.exact_c.committed = 1;
        observations.exact_c.iteration_count_outcomes = 1;
        observations.exact_c.attempted_iterations = 4;
        observations.exact_c.accepted_iterations = 2;
        observations.exact_c.multi_iteration_evidence_outcomes = 1;
        observations.exact_c.multiplicative_weights_requested = Some(false);
        observations.exact_c.completed_proposals = 4;
        observations.exact_c.gradient_plan_num_specs = Some(1);
        observations.exact_c.gradient_row_count = Some(3);
        observations
            .exact_c
            .stop_reasons
            .insert("iteration_limit".to_string(), 1);
        let rendered = format_competition_json_with_observations(payload, &observations)
            .expect("competition JSON serializes");
        let rendered: serde_json::Value =
            serde_json::from_str(&rendered).expect("competition JSON parses");
        assert_eq!(
            rendered["effective_config"]["schema"],
            EFFECTIVE_TREATMENT_SCHEMA
        );
        assert_eq!(rendered["effective_config"]["clip"]["interm_topk"], 20);
        assert!(
            rendered["effective_config"]
                .get("execution_observations")
                .is_none(),
            "runtime counters must not enter the stable treatment projection"
        );
        assert_eq!(
            rendered["execution_observations"]["schema"],
            "ny_beta_crown_execution_observations_v5"
        );
        assert_eq!(
            rendered["execution_observations"]["exact_c"]["selected_iteration_limit"],
            4
        );
        assert_eq!(
            rendered["execution_observations"]["exact_c"]["attempted_iterations"],
            4
        );
        assert_eq!(
            rendered["execution_observations"]["exact_c"]["multi_iteration_evidence_outcomes"],
            1
        );
        assert_eq!(
            rendered["execution_observations"]["exact_c"]["multiplicative_weights_requested"],
            false
        );
        assert_eq!(
            rendered["effective_config"]["clip"]["input_split_fresh_domain_clip_configured"],
            true
        );
        assert_eq!(
            rendered["execution_observations"]["fresh_domain_clip"]["observed"], false,
            "a configured request alone must not claim runtime dispatch"
        );
        assert_eq!(
            rendered["execution_observations"]["fresh_domain_clip"]["configured"],
            serde_json::Value::Null
        );
        assert_eq!(
            rendered["execution_observations"]["fresh_domain_clip"]["route_authorized"],
            serde_json::Value::Null
        );
        assert_eq!(
            rendered["execution_observations"]["fresh_domain_clip"]["attempts"],
            0
        );
        assert_eq!(
            rendered["execution_observations"]["patches_materialization"]["observed"],
            false
        );
        assert_eq!(
            rendered["execution_observations"]["patches_materialization"]["attempts"],
            0
        );
    }

    #[test]
    fn effective_treatment_reports_typed_depth_two_select_arming() {
        let mut config = BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::Kfsb,
            fsb_candidates: 7,
            kfsb_reduce_op: KfsbReduceOp::Min,
            use_kfsb_multi_branching: false,
            ..BetaCrownConfig::default()
        };
        config.depth_two_branch_lookahead.mode = DepthTwoBranchLookaheadMode::Select;

        let projection = EffectiveTreatmentProjection::from_resolved_with_gates(
            &config,
            true,
            false,
            true,
            false,
            true,
            false,
            CompleteVerifierArg::Auto,
            BackendArg::Wgpu,
            "cpu".to_string(),
            Some(false),
            treatment_gates(&config),
        );
        let json = serde_json::to_value(&projection).expect("projection serializes");
        assert_eq!(json["branching"]["depth_two_lookahead_mode"], "select");
        assert_eq!(
            json["branching"]["depth_two_lookahead_round_zero_supported"],
            true
        );
        assert_eq!(json["branching"]["kfsb_multi_configured"], false);
        assert_eq!(json["branching"]["wave_kfsb_armed"], true);

        let mut killed_gates = treatment_gates(&config);
        killed_gates.kfsb_multi_env_override = Some(false);
        let killed = EffectiveTreatmentProjection::from_resolved_with_gates(
            &config,
            true,
            false,
            true,
            false,
            true,
            false,
            CompleteVerifierArg::Auto,
            BackendArg::Wgpu,
            "cpu".to_string(),
            Some(false),
            killed_gates,
        );
        let killed_json = serde_json::to_value(&killed).expect("projection serializes");
        assert_eq!(killed_json["branching"]["wave_kfsb_armed"], false);
    }

    #[test]
    fn effective_treatment_reports_legacy_adaptive_depth_gates_separately() {
        let mut config = BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::Kfsb,
            fsb_candidates: 7,
            kfsb_reduce_op: KfsbReduceOp::Min,
            ..BetaCrownConfig::default()
        };
        config.depth_two_branch_lookahead.mode = DepthTwoBranchLookaheadMode::Select;

        ny_test_utils::env::with_serialized_env_vars(
            &[
                ("NY_MO_ADAPTIVE_DEPTH_SHADOW", "1"),
                ("NY_MO_ADAPTIVE_DEPTH_SELECT", "0"),
                ("NY_MO_ADAPTIVE_DEPTH_COMMIT", "1"),
                ("NY_MO_KFSB_F64_SHADOW", "0"),
            ],
            || {
                let gates = RuntimeTreatmentGates::capture(&config);
                assert!(gates.adaptive_depth_shadow_env_armed);
                assert!(!gates.adaptive_depth_select_env_armed);
                assert!(gates.adaptive_depth_commit_env_armed);
                assert!(gates.depth_two_lookahead_legacy_observer_conflict);

                let projection = EffectiveTreatmentProjection::from_resolved_with_gates(
                    &config,
                    true,
                    false,
                    true,
                    false,
                    true,
                    false,
                    CompleteVerifierArg::Auto,
                    BackendArg::Wgpu,
                    "cpu".to_string(),
                    Some(false),
                    gates,
                );
                let json = serde_json::to_value(&projection).expect("projection serializes");
                assert_eq!(json["branching"]["adaptive_depth_shadow_env_armed"], true);
                assert_eq!(json["branching"]["adaptive_depth_select_env_armed"], false);
                assert_eq!(json["branching"]["adaptive_depth_commit_env_armed"], true);
                assert_eq!(
                    json["branching"]["depth_two_lookahead_legacy_observer_conflict"],
                    true
                );
                assert_eq!(
                    json["branching"]["depth_two_lookahead_round_zero_supported"],
                    false,
                    "legacy COMMIT must conflict with the typed depth-two lane just as the solver does"
                );
            },
        );

        ny_test_utils::env::with_serialized_env_vars(
            &[
                ("NY_MO_ADAPTIVE_DEPTH_SHADOW", "true"),
                ("NY_MO_ADAPTIVE_DEPTH_SELECT", "01"),
                ("NY_MO_ADAPTIVE_DEPTH_COMMIT", " 1"),
                ("NY_MO_KFSB_F64_SHADOW", "0"),
            ],
            || {
                let gates = RuntimeTreatmentGates::capture(&config);
                assert!(!gates.adaptive_depth_shadow_env_armed);
                assert!(!gates.adaptive_depth_select_env_armed);
                assert!(!gates.adaptive_depth_commit_env_armed);
                assert!(!gates.depth_two_lookahead_legacy_observer_conflict);
            },
        );
    }

    #[test]
    fn effective_treatment_reports_softmax_objective_envelope_exactly() {
        let config = BetaCrownConfig::default();
        let render = |gates: RuntimeTreatmentGates| {
            let projection = EffectiveTreatmentProjection::from_resolved_with_gates(
                &config,
                true,
                false,
                true,
                false,
                true,
                false,
                CompleteVerifierArg::Auto,
                BackendArg::Cpu,
                "cpu".to_string(),
                Some(false),
                gates,
            );
            serde_json::to_value(&projection).expect("projection serializes")
        };

        ny_test_utils::env::with_serialized_env_vars_removed(
            &["NY_SOFTMAX_OBJECTIVE_ENVELOPE"],
            || {
                let gates = RuntimeTreatmentGates::capture(&config);
                assert!(!gates.softmax_objective_envelope);
                assert_eq!(
                    render(gates)["softmax"]["objective_envelope_env_armed"],
                    false
                );
            },
        );

        for (raw, expected) in [
            ("0", false),
            ("1", true),
            ("", false),
            ("true", false),
            ("01", false),
            (" 1", false),
        ] {
            ny_test_utils::env::with_serialized_env_vars(
                &[("NY_SOFTMAX_OBJECTIVE_ENVELOPE", raw)],
                || {
                    let gates = RuntimeTreatmentGates::capture(&config);
                    assert_eq!(gates.softmax_objective_envelope, expected, "raw={raw:?}");
                    assert_eq!(
                        render(gates)["softmax"]["objective_envelope_env_armed"],
                        expected,
                        "raw={raw:?}"
                    );
                },
            );
        }
    }

    #[test]
    fn effective_treatment_reports_terminal_peel_request_and_application_separately() {
        let config = BetaCrownConfig::default();
        let render = |requested, applied| {
            let projection = EffectiveTreatmentProjection::from_resolved(
                &config,
                true,
                false,
                true,
                false,
                true,
                false,
                CompleteVerifierArg::Auto,
                BackendArg::Cpu,
                "cpu",
                Some(false),
            )
            .with_terminal_peel(requested, applied);
            serde_json::to_value(projection).expect("projection serializes")
        };

        let off = render(false, AppliedTerminalPeel::None);
        assert_eq!(off["softmax"]["terminal_peel_requested"], false);
        assert_eq!(off["softmax"]["terminal_peel_applied"], false);
        assert_eq!(off["softmax"]["terminal_peel_activation"], "none");

        let declined = render(true, AppliedTerminalPeel::None);
        assert_eq!(declined["softmax"]["terminal_peel_requested"], true);
        assert_eq!(declined["softmax"]["terminal_peel_applied"], false);
        assert_eq!(declined["softmax"]["terminal_peel_activation"], "none");

        let applied = render(true, AppliedTerminalPeel::Softmax);
        assert_eq!(applied["softmax"]["terminal_peel_requested"], true);
        assert_eq!(applied["softmax"]["terminal_peel_applied"], true);
        assert_eq!(applied["softmax"]["terminal_peel_activation"], "softmax");
    }

    #[test]
    fn effective_treatment_distinguishes_kfsb_cert_configured_from_armed() {
        let config = BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::Kfsb,
            use_kfsb_multi_branching: true,
            kfsb_cert_reuse: true,
            ..BetaCrownConfig::default()
        };
        let mut gates = treatment_gates(&config);
        gates.kfsb_cert_reuse_armed = false;
        let projection = EffectiveTreatmentProjection::from_resolved_with_gates(
            &config,
            true,
            false,
            true,
            false,
            true,
            false,
            CompleteVerifierArg::Auto,
            BackendArg::Wgpu,
            "cpu".to_string(),
            Some(false),
            gates,
        );
        let json = serde_json::to_value(&projection).expect("projection serializes");
        assert_eq!(json["branching"]["kfsb_cert_reuse_configured"], true);
        assert_eq!(json["branching"]["kfsb_cert_reuse_armed"], false);
    }

    #[test]
    fn runtime_treatment_capture_honors_typed_kfsb_cert_policy_and_kill_switch() {
        let configured = BetaCrownConfig {
            kfsb_cert_reuse: true,
            ..BetaCrownConfig::default()
        };
        ny_test_utils::env::with_serialized_env_vars_removed(&["NY_MO_KFSB_CERT_REUSE"], || {
            assert!(RuntimeTreatmentGates::capture(&configured).kfsb_cert_reuse_armed)
        });
        ny_test_utils::env::with_serialized_env_vars(&[("NY_MO_KFSB_CERT_REUSE", "0")], || {
            assert!(!RuntimeTreatmentGates::capture(&configured).kfsb_cert_reuse_armed)
        });

        let default_dark = BetaCrownConfig::default();
        ny_test_utils::env::with_serialized_env_vars(&[("NY_MO_KFSB_CERT_REUSE", "1")], || {
            assert!(RuntimeTreatmentGates::capture(&default_dark).kfsb_cert_reuse_armed)
        });
    }

    #[test]
    fn effective_treatment_reports_exact_c_and_complete_invprop_ingress() {
        let mut config = BetaCrownConfig {
            atomic_root_c_margin_iterations: 4,
            adv_check: 0,
            ..BetaCrownConfig::default()
        };
        config.alpha_config.invprop.enabled = true;
        config.alpha_config.invprop.apply_output_constraints_to =
            vec!["BoundLinear".to_string(), "/input.7".to_string()];
        config.alpha_config.invprop.tighten_input_bounds = true;
        config.alpha_config.invprop.best_of_oc_and_no_oc = true;
        config.alpha_config.invprop.directly_optimize =
            vec!["/input".to_string(), "head".to_string()];
        config.alpha_config.invprop.share_gammas = true;
        config.alpha_config.invprop.per_layer_gammas = true;
        config.alpha_config.invprop.optimize_gammas = true;
        config.alpha_config.invprop.gamma_lr = 0.125;
        config.alpha_config.output_constraints = Some(
            ny_propagate::OutputConstraints::new(
                arr2(&[[1.0, -1.0], [-1.0, 1.0]]),
                arr1(&[0.0, 0.25]),
                true,
            )
            .expect("well-shaped output-constraint matrix"),
        );

        let mut gates = treatment_gates(&config);
        gates.root_spec_prune_requested = true;
        let projection = EffectiveTreatmentProjection::from_resolved_with_gates(
            &config,
            true,
            false,
            true,
            false,
            true,
            false,
            CompleteVerifierArg::Auto,
            BackendArg::Cpu,
            "cpu".to_string(),
            Some(false),
            gates,
        );
        let json = serde_json::to_value(&projection).expect("projection serializes");

        assert_eq!(json["root"]["atomic_root_c_margin_iterations"], 4);
        assert_eq!(json["root"]["root_spec_prune_requested"], true);
        assert_eq!(json["branching"]["input_split_adv_check"], 0);
        assert_eq!(json["invprop"]["enabled"], true);
        assert_eq!(
            json["invprop"]["apply_output_constraints_to"],
            serde_json::json!(["BoundLinear", "/input.7"])
        );
        assert_eq!(json["invprop"]["tighten_input_bounds"], true);
        assert_eq!(json["invprop"]["best_of_oc_and_no_oc"], true);
        assert_eq!(
            json["invprop"]["directly_optimize"],
            serde_json::json!(["/input", "head"])
        );
        assert_eq!(json["invprop"]["share_gammas"], true);
        assert_eq!(json["invprop"]["per_layer_gammas"], true);
        assert_eq!(json["invprop"]["optimize_gammas"], true);
        assert_eq!(
            json["invprop"]["gamma_lr"],
            serde_json::json!(f64::from(0.125_f32))
        );
        let matrix = &json["invprop"]["top_level_output_constraint_matrix"];
        assert_eq!(matrix["rows"], 2);
        assert_eq!(matrix["columns"], 2);
        assert_eq!(matrix["rhs_entries"], 2);
        assert_eq!(matrix["is_conjunction"], true);
        assert_eq!(matrix["clause_indices_present"], false);
        assert_eq!(matrix["clause_count"], serde_json::Value::Null);
        assert_eq!(matrix["clause_row_counts"], serde_json::Value::Null);
        assert_eq!(
            json["invprop"]["serial_clause_rebinding"],
            "not_applicable_for_top_level_conjunction"
        );
        assert_eq!(json["invprop"]["split_lift_requested"], false);
        assert_eq!(json["invprop"]["split_lift_effective_armed"], false);
    }

    #[test]
    fn effective_treatment_does_not_claim_unobserved_clause_or_split_lift_activation() {
        let mut config = BetaCrownConfig::default();
        config.alpha_config.invprop.enabled = false;
        config.alpha_config.output_constraints = None;
        let mut gates = treatment_gates(&config);
        gates.invprop_split_lift_requested = true;

        let projection = EffectiveTreatmentProjection::from_resolved_with_gates(
            &config,
            true,
            false,
            true,
            false,
            true,
            false,
            CompleteVerifierArg::Auto,
            BackendArg::Cpu,
            "cpu".to_string(),
            Some(true),
            gates,
        );
        let json = serde_json::to_value(&projection).expect("projection serializes");

        assert_eq!(
            json["invprop"]["top_level_output_constraint_matrix"],
            serde_json::Value::Null
        );
        assert_eq!(
            json["invprop"]["serial_clause_rebinding"],
            "possible_but_unobserved_for_top_level_disjunctions"
        );
        assert_eq!(json["invprop"]["split_lift_requested"], true);
        assert_eq!(json["invprop"]["split_lift_effective_armed"], false);
    }

    #[test]
    fn standalone_gate_checks_the_decimal_actually_emitted_for_x() {
        let spec = exact_point_one_tenth_spec();
        let point = 0.1_f32;
        assert_ne!(
            f64::from(point),
            0.1,
            "the f32 promotion differs from the emitted shortest decimal"
        );
        assert_eq!(
            point
                .to_string()
                .parse::<f64>()
                .expect("f32 Display parses"),
            0.1
        );
        assert!(
            standalone_witness_matches_spec(&spec, &[point], &[1.0]),
            "the organizer parses emitted X_0 as the exact declared decimal"
        );

        let widened_upper = f32::from_bits(point.to_bits() + 1);
        assert!(
            !standalone_witness_matches_spec(&spec, &[widened_upper], &[1.0]),
            "an attack-only outward ULP must not cross the publication seam"
        );

        let rendered = format_competition_json(beta_crown_json_payload(
            &violated_result(point),
            None,
            0.0,
            0.0,
            AppliedTerminalPeel::None,
            None,
        ))
        .expect("counterexample JSON");
        let parsed: serde_json::Value =
            serde_json::from_str(&rendered).expect("counterexample JSON parses");
        assert_eq!(
            parsed["counterexample"]["input"][0],
            serde_json::json!(0.1_f64),
            "nested JSON X and counterexample_vnnlib must encode the same decimal"
        );
    }

    #[test]
    fn typed_terminal_peel_repairs_sigmoid_and_refuses_unrehydrated_softmax() {
        let result = BetaCrownResult {
            result: BabVerificationStatus::Violated {
                counterexample: vec![0.25],
                output: vec![0.0],
            },
            domains_explored: 1,
            time_elapsed: Duration::ZERO,
            max_depth_reached: 0,
            output_bounds: None,
            cuts_generated: 0,
            domains_verified: 0,
        };

        let sigmoid =
            beta_crown_json_payload(&result, None, 0.0, 0.0, AppliedTerminalPeel::Sigmoid, None);
        assert_eq!(
            sigmoid
                .counterexample
                .as_ref()
                .map(|(_, output)| &output[..]),
            Some(&[0.5][..]),
            "a relational legacy Sigmoid peel must not publish logits as Y"
        );

        let mut softmax =
            beta_crown_json_payload(&result, None, 0.0, 0.0, AppliedTerminalPeel::Softmax, None);
        assert!(refuse_unrehydrated_softmax_family_witness(
            &mut softmax,
            AppliedTerminalPeel::Softmax,
        ));
        assert_eq!(softmax.status, "unknown");
        assert!(softmax.counterexample.is_none());

        // The in-process traffic route deliberately skips that standalone
        // refusal and retains the input/logit seed for outer ORT rehydration.
        let captured_seed =
            beta_crown_json_payload(&result, None, 0.0, 0.0, AppliedTerminalPeel::Softmax, None);
        assert_eq!(captured_seed.status, "violated");
        assert!(captured_seed.counterexample.is_some());
    }

    #[test]
    fn standalone_gate_enforces_declared_and_same_clause_boxes_at_zero_tolerance() {
        use std::collections::BTreeMap;

        let mut spec = VnnLibSpec::new();
        spec.num_inputs = 2;
        spec.num_outputs = 1;
        spec.input_bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        spec.declared_input_bounds = vec![(0.0, 0.5), (0.0, 0.5)];
        let clause = vec![ny_onnx::vnnlib::OutputConstraint::GreaterEqConst(0, 0.0)];
        spec.output_constraints = clause.clone();
        spec.output_constraint_clauses = vec![clause];
        spec.is_disjunction = true;
        let mut clause_box = BTreeMap::new();
        clause_box.insert(1, (0.0, 0.25));
        spec.per_clause_input_bounds = vec![clause_box];

        assert!(standalone_witness_matches_spec(&spec, &[0.5, 0.25], &[1.0]));
        assert!(
            !standalone_witness_matches_spec(&spec, &[0.500_000_06, 0.25], &[1.0]),
            "declared top-level bounds apply in addition to clause bounds"
        );
        assert!(
            !standalone_witness_matches_spec(&spec, &[0.5, 0.250_000_03], &[1.0]),
            "per-clause membership has organizer-exact zero tolerance"
        );
        assert!(!standalone_witness_matches_spec(
            &spec,
            &[0.5, 0.25],
            &[f32::NAN]
        ));
    }

    #[test]
    fn standalone_invalid_witness_downgrades_but_vnncomp_capture_keeps_refinement_seed() {
        let mut property = tempfile::NamedTempFile::new().expect("temporary VNN-LIB");
        write!(
            property,
            "(declare-const X_0 Real)\n\
             (declare-const Y_0 Real)\n\
             (assert (>= X_0 0.1))\n\
             (assert (<= X_0 0.1))\n\
             (assert (>= Y_0 0.0))\n"
        )
        .expect("write VNN-LIB");

        let widened_upper = f32::from_bits(0.1_f32.to_bits() + 1);
        let result = violated_result(widened_upper);
        let mut standalone = beta_crown_json_payload(
            &result,
            Some(property.path()),
            0.0,
            0.0,
            AppliedTerminalPeel::None,
            None,
        );
        // No ONNX model in this fixture, so the ORT re-execution gate cannot run and
        // the pre-existing organizer-exact spec check is what decides — which is
        // exactly what this test pins.
        seal_standalone_witness(&mut standalone, Some(property.path()), None);
        assert_eq!(standalone.status, "unknown");
        assert!(standalone.counterexample.is_none());

        // Captured JSON is an INTERNAL candidate, not the final VNN-COMP
        // witness. Preserve it so the trusted ORT gate can snap/refine the
        // non-f32 pinned value before organizer publication.
        begin_capture();
        output_result(
            &result,
            &Some(property.path().to_path_buf()),
            None,
            0.0,
            0.0,
            true,
            true,
            false,
            &default_treatment(),
        )
        .expect("captured rendering");
        let captured = take_captured_json().expect("captured JSON");
        end_capture();
        let parsed: serde_json::Value =
            serde_json::from_str(&captured).expect("captured JSON parses");
        assert_eq!(parsed["status"], "violated");
        assert!(parsed["counterexample"].is_object());
    }

    #[test]
    fn standalone_bab_publication_preserves_strictness_and_exact_box_membership() {
        let write_property = |comparison: &str| {
            let mut property = tempfile::NamedTempFile::new().expect("temporary VNN-LIB");
            write!(
                property,
                "(declare-const X_0 Real)\n\
                 (declare-const Y_0 Real)\n\
                 (assert (>= X_0 0.0))\n\
                 (assert (<= X_0 1.0))\n\
                 (assert ({comparison} Y_0 0.0))\n"
            )
            .expect("write VNN-LIB");
            property
        };
        let strict = write_property(">");
        let non_strict = write_property(">=");
        let mut boundary = violated_result(0.5);
        boundary.result = BabVerificationStatus::Violated {
            counterexample: vec![0.5],
            output: vec![0.0],
        };

        assert!(
            !standalone_bab_witness_is_publishable(
                &boundary,
                Some(strict.path()),
                AppliedTerminalPeel::None,
            ),
            "equality does not satisfy an authored strict unsafe atom"
        );
        assert!(
            standalone_bab_witness_is_publishable(
                &boundary,
                Some(non_strict.path()),
                AppliedTerminalPeel::None,
            ),
            "equality is a genuine violation for the non-strict sibling"
        );

        let mut outside_box = boundary;
        if let BabVerificationStatus::Violated { counterexample, .. } = &mut outside_box.result {
            *counterexample = vec![f32::from_bits(1.0_f32.to_bits() + 1)];
        }
        assert!(
            !standalone_bab_witness_is_publishable(
                &outside_box,
                Some(non_strict.path()),
                AppliedTerminalPeel::None,
            ),
            "an outward-rounded shell point must not escape through human output"
        );
    }

    #[test]
    fn test_format_beta_crown_json_nests_counterexample_3708() {
        let payload = beta_crown_json_payload(
            &BetaCrownResult {
                result: BabVerificationStatus::Violated {
                    counterexample: vec![0.5],
                    output: vec![-1.0],
                },
                domains_explored: 3,
                time_elapsed: Duration::from_secs(2),
                max_depth_reached: 1,
                output_bounds: None,
                cuts_generated: 0,
                domains_verified: 1,
            },
            None,
            0.02,
            0.0,
            AppliedTerminalPeel::None,
            None,
        );
        let json =
            format_competition_json(payload).expect("beta-crown JSON payload should serialize");
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("beta-crown JSON must parse");

        assert_eq!(parsed["status"], "violated");
        assert_eq!(parsed["counterexample"]["input"][0], 0.5);
        assert_eq!(parsed["counterexample"]["output"][0], -1.0);
        assert!(parsed.get("output").is_none(), "output must stay nested");
    }

    #[test]
    fn test_counterexample_vnnlib_smtlib_format() {
        // Synthetic 2-input / 2-output counterexample.
        let input = [0.123_456_f32, -0.654_321_f32];
        let output = [0.1_f32, -0.3_f32];
        let smt = counterexample_vnnlib(&input, &output);

        // Outer parentheses wrap the whole assignment list.
        assert!(smt.starts_with('('), "must start with '(': {smt}");
        assert!(smt.ends_with(')'), "must end with ')': {smt}");

        // One assignment per line; inputs are X_i, outputs are Y_j, in order.
        let lines: Vec<&str> = smt.lines().collect();
        assert_eq!(lines.len(), 4, "one assignment per line: {smt}");
        assert!(lines[0].starts_with("((X_0 "), "first input X_0: {smt}");
        assert!(lines[1].starts_with("(X_1 "), "second input X_1: {smt}");
        assert!(lines[2].starts_with("(Y_0 "), "first output Y_0: {smt}");
        assert!(lines[3].starts_with("(Y_1 "), "second output Y_1: {smt}");

        // Each line is a balanced parenthesised assignment.
        for line in &lines {
            assert!(line.contains('('), "assignment has '(': {line}");
            assert!(line.ends_with(')'), "assignment ends ')': {line}");
        }

        // Full precision retained (not truncated to 6 decimals).
        assert!(smt.contains("0.123456"), "input precision preserved: {smt}");
        assert!(
            smt.contains("-0.654321"),
            "negative input precision preserved: {smt}"
        );

        // X formatting is intentionally unchanged: replay must see the exact
        // same f32 tensor as before this Y-only compatibility hardening.
        assert_eq!(lines[0], format!("((X_0 {})", input[0]));
        assert_eq!(lines[1], format!("(X_1 {})", input[1]));

        // Y text round-trips through the checker's f64 parser to the exact f64
        // promotion of the ONNX f32.  Plain f32 Display does not have that
        // property for 0.1 and was classified tolerance-only by the 2025
        // zero-tolerance checker.
        let y0_text = lines[2]
            .strip_prefix("(Y_0 ")
            .and_then(|text| text.strip_suffix(')'))
            .expect("Y_0 assignment");
        let y0: f64 = y0_text.parse().expect("Y_0 decimal");
        assert_eq!(y0, f64::from(output[0]));
        assert_ne!(
            output[0]
                .to_string()
                .parse::<f64>()
                .expect("short f32 decimal"),
            f64::from(output[0])
        );
    }

    #[test]
    fn test_format_beta_crown_json_includes_vnnlib_witness() {
        let payload = beta_crown_json_payload(
            &BetaCrownResult {
                result: BabVerificationStatus::Violated {
                    counterexample: vec![0.5, -0.25],
                    output: vec![-1.0, 2.0],
                },
                domains_explored: 1,
                time_elapsed: Duration::from_secs(1),
                max_depth_reached: 1,
                output_bounds: None,
                cuts_generated: 0,
                domains_verified: 0,
            },
            None,
            0.02,
            0.0,
            AppliedTerminalPeel::None,
            None,
        );
        let json =
            format_competition_json(payload).expect("beta-crown JSON payload should serialize");
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("beta-crown JSON must parse");

        let witness = parsed["counterexample_vnnlib"]
            .as_str()
            .expect("counterexample_vnnlib must be a string");
        assert!(witness.contains("(X_0 "), "witness has X_0: {witness}");
        assert!(witness.contains("(X_1 "), "witness has X_1: {witness}");
        assert!(witness.contains("(Y_0 "), "witness has Y_0: {witness}");
        assert!(witness.contains("(Y_1 "), "witness has Y_1: {witness}");
    }

    #[test]
    fn test_format_beta_crown_json_omits_vnnlib_when_no_counterexample() {
        let payload = beta_crown_json_payload(
            &BetaCrownResult {
                result: BabVerificationStatus::Verified,
                domains_explored: 1,
                time_elapsed: Duration::from_secs(1),
                max_depth_reached: 0,
                output_bounds: None,
                cuts_generated: 0,
                domains_verified: 1,
            },
            None,
            0.02,
            0.0,
            AppliedTerminalPeel::None,
            None,
        );
        let json = format_competition_json(payload).expect("verified JSON should serialize");
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("verified JSON must parse");
        assert!(
            parsed["counterexample_vnnlib"].is_null(),
            "no witness when verified: {json}"
        );
    }

    #[test]
    fn test_potential_violation_exit_code_is_unknown_3708() {
        assert_eq!(
            beta_crown_exit_code(&BabVerificationStatus::potential_violation()),
            exit_codes::UNKNOWN
        );
    }

    // Tests for the SMT/MIP-specific format_verification_result_json are gated
    // behind the smt/mip features since the function itself is cfg-gated.
    #[cfg(feature = "mip")]
    mod verification_result_json_tests {
        use super::super::*;
        use ny_core::{Bound, SoundnessProvenance, UnknownReason, VerificationResult};
        use std::io::Write;
        use std::time::Duration;

        fn violated_verification_result() -> VerificationResult {
            VerificationResult::Violated {
                provenance: SoundnessProvenance::sound(),
                counterexample: vec![0.25, 0.75],
                output: vec![1.0, -2.0],
                details: None,
                actual_method: Some(ny_core::MethodUsed::MipHiGHS),
            }
        }

        #[test]
        fn test_format_verification_result_json_nests_counterexample_3708() {
            let config = BetaCrownConfig::default();
            let effective = EffectiveTreatmentProjection::from_resolved(
                &config,
                false,
                false,
                false,
                false,
                false,
                false,
                CompleteVerifierArg::Mip,
                BackendArg::Cpu,
                "cpu",
                None,
            );
            let json = format_verification_result_json(
                &violated_verification_result(),
                None,
                0.01,
                0.0,
                Duration::from_millis(1500),
                "mip-highs",
                Some(&effective),
            )
            .expect("verification result JSON should serialize");

            assert!(
                json.contains("\"status\": \"violated\""),
                "status field should match run_instance grep contract: {json}"
            );
            assert!(
                !json.contains("\\\"status\\\""),
                "status field must not be backslash-escaped: {json}"
            );

            let parsed: serde_json::Value =
                serde_json::from_str(&json).expect("verification result JSON must parse");
            assert_eq!(parsed["status"], "violated");
            assert_eq!(parsed["method"], "mip-highs");
            assert_eq!(
                parsed["effective_config"]["schema"],
                EFFECTIVE_TREATMENT_SCHEMA
            );
            assert_eq!(parsed["domains_explored"], 0);
            assert!(parsed.get("output").is_none(), "output must stay nested");
            assert_eq!(parsed["counterexample"]["input"][0], 0.25);
            assert_eq!(parsed["counterexample"]["output"][1], -2.0);
        }

        #[test]
        fn test_format_verification_result_json_preserves_structured_reason_3708() {
            let json = format_verification_result_json(
                &VerificationResult::Unknown {
                    provenance: SoundnessProvenance::sound(),
                    bounds: vec![Bound::new(0.0, 1.0)],
                    reason: UnknownReason::SmtUnknown {
                        solver_reason: Some("ay returned unknown".to_string()),
                    },
                    actual_method: Some(ny_core::MethodUsed::Mip),
                },
                None,
                0.01,
                0.0,
                Duration::from_secs(1),
                "mip",
                None,
            )
            .expect("unknown JSON should serialize");
            let parsed: serde_json::Value =
                serde_json::from_str(&json).expect("unknown JSON must parse");

            assert_eq!(parsed["status"], "unknown");
            assert_eq!(parsed["reason"]["type"], "smt_unknown");
            assert_eq!(parsed["reason"]["solver_reason"], "ay returned unknown");
        }

        #[test]
        fn mip_publication_gate_reports_refusal_and_capture_preserves_seed() {
            let mut property = tempfile::NamedTempFile::new().expect("temporary VNN-LIB");
            write!(
                property,
                "(declare-const X_0 Real)\n\
                 (declare-const Y_0 Real)\n\
                 (assert (>= X_0 0.1))\n\
                 (assert (<= X_0 0.1))\n\
                 (assert (>= Y_0 0.0))\n"
            )
            .expect("write VNN-LIB");
            let result = VerificationResult::Violated {
                provenance: SoundnessProvenance::sound(),
                counterexample: vec![f32::from_bits(0.1_f32.to_bits() + 1)],
                output: vec![1.0],
                details: None,
                actual_method: Some(ny_core::MethodUsed::Mip),
            };

            let (standalone, refused) = format_verification_result_json_for_publication(
                &result,
                Some(property.path()),
                None,
                0.0,
                0.0,
                Duration::ZERO,
                "mip-ay",
                None,
            )
            .expect("standalone MIP JSON");
            assert!(refused, "caller must map this result to Unknown exit");
            let standalone: serde_json::Value =
                serde_json::from_str(&standalone).expect("standalone JSON parses");
            assert_eq!(standalone["status"], "unknown");
            assert!(standalone["counterexample"].is_null());

            begin_capture();
            let (captured, refused) = format_verification_result_json_for_publication(
                &result,
                Some(property.path()),
                None,
                0.0,
                0.0,
                Duration::ZERO,
                "mip-ay",
                None,
            )
            .expect("captured MIP JSON");
            end_capture();
            assert!(!refused, "internal capture keeps the refinement seed");
            let captured: serde_json::Value =
                serde_json::from_str(&captured).expect("captured JSON parses");
            assert_eq!(captured["status"], "violated");
        }

        #[test]
        fn mip_terminal_peel_witness_policy_uses_typed_activation() {
            let config = BetaCrownConfig::default();
            let treatment = |activation| {
                EffectiveTreatmentProjection::from_resolved(
                    &config,
                    false,
                    false,
                    false,
                    false,
                    false,
                    false,
                    CompleteVerifierArg::Mip,
                    BackendArg::Cpu,
                    "cpu",
                    None,
                )
                .with_terminal_peel(true, activation)
            };
            let result = violated_verification_result();

            let softmax = treatment(AppliedTerminalPeel::Softmax);
            let (standalone, refused) = format_verification_result_json_for_publication(
                &result,
                None,
                None,
                0.0,
                0.0,
                Duration::ZERO,
                "mip-ay",
                Some(&softmax),
            )
            .expect("standalone peeled MIP JSON");
            assert!(refused);
            let standalone: serde_json::Value =
                serde_json::from_str(&standalone).expect("standalone JSON parses");
            assert_eq!(standalone["status"], "unknown");

            begin_capture();
            let (captured, refused) = format_verification_result_json_for_publication(
                &result,
                None,
                None,
                0.0,
                0.0,
                Duration::ZERO,
                "mip-ay",
                Some(&softmax),
            )
            .expect("captured peeled MIP JSON");
            end_capture();
            assert!(!refused);
            let captured: serde_json::Value =
                serde_json::from_str(&captured).expect("captured JSON parses");
            assert_eq!(captured["status"], "violated");
            assert_eq!(captured["counterexample"]["output"][0], 1.0);

            let sigmoid = treatment(AppliedTerminalPeel::Sigmoid);
            let rendered = format_verification_result_json(
                &result,
                None,
                0.0,
                0.0,
                Duration::ZERO,
                "mip-ay",
                Some(&sigmoid),
            )
            .expect("Sigmoid-repaired MIP JSON");
            let rendered: serde_json::Value =
                serde_json::from_str(&rendered).expect("Sigmoid JSON parses");
            let expected = AppliedTerminalPeel::Sigmoid
                .output_in_original_coordinates(&[1.0, -2.0])
                .expect("Sigmoid mapping")
                .into_owned();
            assert_eq!(
                rendered["counterexample"]["output"][0],
                serde_json::json!(f64::from(expected[0]))
            );
        }
    }
}
