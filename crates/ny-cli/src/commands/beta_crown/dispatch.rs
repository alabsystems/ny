// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verification dispatch helpers for `ny beta-crown`.
//!
//! Part of Slice 3 for #4246: keep the top-level CLI handler focused on
//! argument/config assembly while this module owns the MIP-only and BaB
//! execution paths.

use anyhow::Result;
use ny_core::GemmEngine;
use ny_gpu::ComputeDevice;
#[cfg(feature = "mip")]
use ny_onnx::load_onnx_with_config;
#[cfg(feature = "mip")]
use ny_onnx::vnnlib::OutputConstraint;
use ny_onnx::{vnnlib::VnnLibSpec, OnnxLoadConfig};
use ny_propagate::{BabVerificationStatus, BetaCrownConfig, BetaCrownResult, BetaCrownVerifier};
use ny_tensor::BoundedTensor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tracing::info;
#[cfg(feature = "mip")]
use tracing::warn;

#[cfg(feature = "mip")]
use super::apply_heuristic_sound_modes;
use super::domain_batch_metrics::JsonlGraphDomainBatchMetricsSink;
use super::ibp_check::ibp_check_vnnlib_safe;
use super::input_split_metrics::JsonlInputSplitMetricsSink;
use super::output::{output_result, EffectiveTreatmentProjection};
use super::verify::phase_budget::PhaseBudgetLedger;
use super::verify::{
    confirm_potential_violation, normalize_result_wall_time, try_pgd_before_mip,
    verify_relational_constraints_with_ledger, verify_standard,
};
use super::{mip_highs, BetaCrownModel};
use crate::{CompleteVerifierArg, MipSolverArg};

/// Conservative cap on total network parameters for auto-escalation to MIP.
///
/// Big-M MIP with one binary per unstable ReLU only stays tractable for small
/// sequential nets (the categories the hand-tuned routing targets — sat_relu,
/// safenlp, malbeware — are all well under this). Above the cap we keep the BaB
/// `unknown`/`timeout` verdict rather than launch a MIP that will itself just
/// time out: escalation must never *cost* us a result we could otherwise report
/// sooner, and never changes a *decided* verdict.
#[cfg(feature = "mip")]
pub(super) const MIP_AUTO_ESCALATION_MAX_PARAMS: usize = 5_000_000;

/// Pure layer-type predicate: is this layer one the feed-forward MIP path can encode
/// (directly, after unfolding Conv2d, or after folding into a neighbour)?
///
/// Mirrors the accepted set in `mip_highs::verify_with_mip` /
/// `mip_preprocess` (`unfold_conv2d_to_linear` + `strip_shape_layers` +
/// `fold_constant_layers`): Linear and ReLU are encoded directly; Conv2d is
/// unfolded to Linear; Flatten/Reshape are stripped; AddConstant and
/// (non-reverse) SubConstant fold into an adjacent Linear bias.
///
/// Anything else (other activations, attention, normalization, binary ops)
/// would make `verify_with_mip` bail, so we must NOT escalate for it.
#[cfg(feature = "mip")]
fn is_mip_encodable_layer(layer: &ny_propagate::Layer) -> bool {
    use ny_propagate::Layer;
    match layer {
        Layer::Linear(_) | Layer::ReLU(_) | Layer::Conv2d(_) => true,
        // Shape-only and constant-fold layers are removed by the MIP
        // preprocessing pipeline before the Linear+ReLU validation runs.
        Layer::Flatten(_) | Layer::Reshape(_) | Layer::AddConstant(_) => true,
        Layer::SubConstant(sub) => !sub.reverse,
        _ => false,
    }
}

/// Decide whether a sequential network is MIP-encodable within the size cap.
///
/// Conservative by construction: returns `true` only when *every* layer is in
/// the encodable set and the total parameter count is under the cap. A `false`
/// result means "do not escalate" — the BaB verdict is reported as-is.
#[cfg(feature = "mip")]
pub(super) fn is_mip_encodable_sequential(
    network: &ny_propagate::Network,
    max_params: usize,
) -> bool {
    use ny_propagate::Layer;
    let mut total_params: usize = 0;
    for layer in network.layers() {
        if !is_mip_encodable_layer(layer) {
            return false;
        }
        if let Layer::Linear(linear) = layer {
            let (out_dim, in_dim) = linear.weight().dim();
            total_params = total_params.saturating_add(out_dim.saturating_mul(in_dim));
        }
    }
    total_params <= max_params
}

/// Decide whether a GRAPH network is MIP-encodable within a BINARY-COUNT
/// budget (the Graph-MIP escalation eligibility check, increment 5).
///
/// Unlike the sequential parameter cap, the binding cost of a big-M MIP is the
/// number of BINARIES — one per UNSTABLE ReLU neuron under the sound
/// per-property bounds (`node_bounds`, flattened; cifar100 root measures ~494
/// under α-CROWN). Conservative by construction: `false` for any layer outside
/// the graph encoder's set (Linear / Conv2d / ReLU / Flatten / Reshape /
/// BatchNorm / Add), for any ReLU with no pre-activation box (the encoder
/// would bail), and above the binary budget.
#[cfg(feature = "mip")]
pub(super) fn is_mip_encodable_graph(
    graph: &ny_propagate::GraphNetwork,
    node_bounds: &std::collections::HashMap<String, Vec<ny_core::Bound>>,
    max_binaries: usize,
) -> bool {
    matches!(
        graph_unstable_relu_count(graph, node_bounds),
        Some(unstable) if unstable <= max_binaries
    )
}

/// #deadlane — is the whole-net Graph-MIP escalation STATICALLY impossible for
/// this model? Used to disarm the BaB deadline's MIP reservation
/// ([`super::verify::phase_budget::PhaseBudgetLedger::with_static_mip_ineligibility`]).
///
/// Answers `true` ONLY on a graph model whose layer set is outside the encoder's
/// set. A sequential model is left alone (it has its own parameter-cap route,
/// [`is_mip_encodable_sequential`]), and a non-mip build never arms the
/// reservation in the first place — both return `false`, i.e. "reserve as before".
pub(super) fn graph_mip_statically_ineligible(model: &BetaCrownModel) -> bool {
    match model {
        BetaCrownModel::Graph(graph) => graph_mip_layer_set_statically_unsupported(graph),
        BetaCrownModel::Sequential(_) => false,
    }
}

/// Exact-string resolver shared with the NY-MIP SafeNLP canary.
///
/// Only literal Unicode `1` arms the budget half of the existing experiment.
/// `0`, whitespace, malformed values, and non-Unicode environment data retain
/// the historical schedule.
#[cfg(feature = "mip")]
fn safenlp_shared_prefix_budget_repair_enabled_from_value(value: Option<&std::ffi::OsStr>) -> bool {
    value == Some(std::ffi::OsStr::new("1"))
}

fn safenlp_shared_prefix_budget_repair_enabled() -> bool {
    #[cfg(feature = "mip")]
    {
        safenlp_shared_prefix_budget_repair_enabled_from_value(
            std::env::var_os("NY_MIP_SAFENLP_SHARED_PREFIX").as_deref(),
        )
    }
    #[cfg(not(feature = "mip"))]
    {
        false
    }
}

/// The direct-MIP-first experiment must preserve the shared-prefix backend's
/// cold-start admission. Historical MIP-only callers retain their configured
/// PGD precheck and optional warm seed; only the typed direct-first ingress is
/// cold by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MipOnlySeedPolicy {
    HistoricalConfigured,
    #[cfg(feature = "mip")]
    ColdNoWarmStart,
}

impl MipOnlySeedPolicy {
    fn run_internal_pgd(self, configured: bool) -> bool {
        matches!(self, Self::HistoricalConfigured) && configured
    }

    #[cfg(feature = "mip")]
    fn permits_warm_start(self) -> bool {
        matches!(self, Self::HistoricalConfigured)
    }
}

#[cfg(feature = "mip")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MipOnlySolverRoute {
    SharedBinaryPrefix,
}

/// Fully admitted direct-first routing decision.
///
/// The caller-owned absolute deadline is copied only as immutable evidence for
/// tests/telemetry. Execution still reads the identical `ctx.overall_deadline`
/// in `run_mip_only_with_seed_policy`; no relative timeout is reconstructed.
#[cfg(feature = "mip")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SafeNlpDirectMipFirstPlan {
    deadline: Instant,
    hidden_dim: usize,
    seed_policy: MipOnlySeedPolicy,
    solver_route: MipOnlySolverRoute,
}

#[cfg(feature = "mip")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SafeNlpDirectMipModelSource {
    Sequential,
    Graph,
}

#[cfg(feature = "mip")]
impl SafeNlpDirectMipModelSource {
    fn from_model(model: &BetaCrownModel) -> Self {
        match model {
            BetaCrownModel::Sequential(_) => Self::Sequential,
            BetaCrownModel::Graph(_) => Self::Graph,
        }
    }

    fn telemetry_name(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Graph => "graph",
        }
    }
}

/// A pre-route decline is deliberately verdict-neutral. Every variant occurs
/// before `route-start`, so production keeps the untouched historical BaB
/// path and may later use its existing post-BaB escalation reload.
#[cfg(feature = "mip")]
#[derive(Debug, Clone, PartialEq, Eq)]
enum SafeNlpDirectMipFirstDecline {
    NotRequested,
    SharedPrefixDisabled,
    NonAutoPolicy,
    NonAyBackend,
    MissingDeadline,
    ExpiredDeadline,
    MissingSpec,
    UnsupportedPropertyShape,
    ReloadFailed(String),
    UnsupportedReloadedModelShape,
}

#[cfg(feature = "mip")]
impl SafeNlpDirectMipFirstDecline {
    fn telemetry(&self) -> (&'static str, &'static str) {
        match self {
            Self::NotRequested => ("preflight", "not-requested"),
            Self::SharedPrefixDisabled => ("preflight", "shared-prefix-disabled"),
            Self::NonAutoPolicy => ("preflight", "non-auto-policy"),
            Self::NonAyBackend => ("preflight", "non-ay-backend"),
            Self::MissingDeadline => ("preflight", "missing-deadline"),
            Self::ExpiredDeadline => ("deadline", "expired"),
            Self::MissingSpec => ("preflight", "missing-spec"),
            Self::UnsupportedPropertyShape => ("admission", "property-shape"),
            Self::ReloadFailed(_) => ("reload", "load-error"),
            Self::UnsupportedReloadedModelShape => ("admission", "reloaded-model-shape"),
        }
    }
}

#[cfg(feature = "mip")]
enum SafeNlpDirectMipFirstAttempt<T> {
    Declined(SafeNlpDirectMipFirstDecline),
    Executed(T),
}

/// Return the unique relational row encoded by the direct conjunctive MIP
/// path, rejecting every richer VNN-LIB shape.
#[cfg(feature = "mip")]
fn safenlp_single_relational_unsafe_row(spec: &VnnLibSpec) -> Option<&OutputConstraint> {
    if spec.is_disjunction || spec.dual_network.is_some() {
        return None;
    }

    let constraints: &[OutputConstraint] = if spec.output_constraint_clauses.is_empty() {
        &spec.output_constraints
    } else {
        let [clause] = spec.output_constraint_clauses.as_slice() else {
            return None;
        };
        if !spec.output_constraints.is_empty()
            && spec.output_constraints.as_slice() != clause.as_slice()
        {
            return None;
        }
        clause
    };
    let [constraint] = constraints else {
        return None;
    };
    let (left, right) = match constraint {
        OutputConstraint::LessEq(left, right)
        | OutputConstraint::GreaterEq(left, right)
        | OutputConstraint::LessThan(left, right)
        | OutputConstraint::GreaterThan(left, right) => (*left, *right),
        _ => return None,
    };
    (left != right && left < spec.num_outputs && right < spec.num_outputs).then_some(constraint)
}

#[cfg(feature = "mip")]
fn safenlp_direct_mip_first_preflight(
    requested_by_vnncomp: bool,
    shared_prefix_enabled: bool,
    complete_verifier: CompleteVerifierArg,
    mip_solver: MipSolverArg,
    spec: Option<&VnnLibSpec>,
    deadline: Option<Instant>,
    now: Instant,
) -> std::result::Result<(Instant, &VnnLibSpec), SafeNlpDirectMipFirstDecline> {
    if !requested_by_vnncomp {
        return Err(SafeNlpDirectMipFirstDecline::NotRequested);
    }
    if !shared_prefix_enabled {
        return Err(SafeNlpDirectMipFirstDecline::SharedPrefixDisabled);
    }
    if complete_verifier != CompleteVerifierArg::Auto {
        return Err(SafeNlpDirectMipFirstDecline::NonAutoPolicy);
    }
    if mip_solver.mip_backend() != ny_mip::MipBackend::Ay {
        return Err(SafeNlpDirectMipFirstDecline::NonAyBackend);
    }
    let deadline = deadline.ok_or(SafeNlpDirectMipFirstDecline::MissingDeadline)?;
    if now >= deadline {
        return Err(SafeNlpDirectMipFirstDecline::ExpiredDeadline);
    }
    let spec = spec.ok_or(SafeNlpDirectMipFirstDecline::MissingSpec)?;
    safenlp_single_relational_unsafe_row(spec)
        .ok_or(SafeNlpDirectMipFirstDecline::UnsupportedPropertyShape)?;
    Ok((deadline, spec))
}

#[cfg(feature = "mip")]
fn safenlp_direct_mip_first_plan_for_reloaded_model(
    model: &BetaCrownModel,
    input: &BoundedTensor,
    spec: &VnnLibSpec,
    deadline: Instant,
    now: Instant,
) -> std::result::Result<SafeNlpDirectMipFirstPlan, SafeNlpDirectMipFirstDecline> {
    if now >= deadline {
        return Err(SafeNlpDirectMipFirstDecline::ExpiredDeadline);
    }
    let hidden_dim = mip_highs::safenlp_canonical_single_hidden_shape(model, input, spec)
        .ok()
        .flatten()
        .ok_or(SafeNlpDirectMipFirstDecline::UnsupportedReloadedModelShape)?;
    Ok(SafeNlpDirectMipFirstPlan {
        deadline,
        hidden_dim,
        seed_policy: MipOnlySeedPolicy::ColdNoWarmStart,
        solver_route: MipOnlySolverRoute::SharedBinaryPrefix,
    })
}

/// Run the pre-route reload and, only after every gate succeeds, invoke one
/// caller-owned route executor.
///
/// The injected closures make the one-reload/one-execution contract directly
/// testable without touching the filesystem or starting AY. `now_after_reload`
/// is sampled both after loading and after structural admission so `route-start`
/// can never be emitted for work that consumed the immutable deadline.
#[cfg(feature = "mip")]
#[allow(clippy::too_many_arguments)]
fn safenlp_direct_mip_first_attempt_with_reload<T, Reload, Now, Execute>(
    requested_by_vnncomp: bool,
    shared_prefix_enabled: bool,
    complete_verifier: CompleteVerifierArg,
    mip_solver: MipSolverArg,
    source: SafeNlpDirectMipModelSource,
    input: &BoundedTensor,
    spec: Option<&VnnLibSpec>,
    deadline: Option<Instant>,
    now_before_reload: Instant,
    reload: Reload,
    mut now_after_reload: Now,
    execute: Execute,
) -> SafeNlpDirectMipFirstAttempt<T>
where
    Reload: FnOnce() -> Result<BetaCrownModel>,
    Now: FnMut() -> Instant,
    Execute: FnOnce(BetaCrownModel, SafeNlpDirectMipFirstPlan, SafeNlpDirectMipModelSource) -> T,
{
    let (deadline, spec) = match safenlp_direct_mip_first_preflight(
        requested_by_vnncomp,
        shared_prefix_enabled,
        complete_verifier,
        mip_solver,
        spec,
        deadline,
        now_before_reload,
    ) {
        Ok(preflight) => preflight,
        Err(reason) => return SafeNlpDirectMipFirstAttempt::Declined(reason),
    };
    let model = match reload() {
        Ok(model) => model,
        Err(error) => {
            return SafeNlpDirectMipFirstAttempt::Declined(
                SafeNlpDirectMipFirstDecline::ReloadFailed(format!("{error:#}")),
            )
        }
    };
    let plan = match safenlp_direct_mip_first_plan_for_reloaded_model(
        &model,
        input,
        spec,
        deadline,
        now_after_reload(),
    ) {
        Ok(plan) => plan,
        Err(reason) => return SafeNlpDirectMipFirstAttempt::Declined(reason),
    };
    if now_after_reload() >= deadline {
        return SafeNlpDirectMipFirstAttempt::Declined(
            SafeNlpDirectMipFirstDecline::ExpiredDeadline,
        );
    }
    SafeNlpDirectMipFirstAttempt::Executed(execute(model, plan, source))
}

/// Exact fresh sequential seam shared in shape with historical post-BaB MIP
/// escalation. One ONNX load produces one `Network`, it is wrapped once as a
/// sequential β-CROWN model, and the existing heuristic sound-mode policy is
/// applied once before structural admission.
#[cfg(feature = "mip")]
fn reload_safenlp_direct_mip_model(
    model_path: &Path,
    onnx_load_config: &OnnxLoadConfig,
    allow_heuristic_logsoftmax: bool,
    allow_heuristic_softmax: bool,
    json: bool,
) -> Result<BetaCrownModel> {
    let seq_net = load_onnx_with_config(model_path, onnx_load_config)
        .map_err(anyhow::Error::from)
        .and_then(|onnx_reload| {
            onnx_reload
                .to_propagate_network()
                .map_err(anyhow::Error::from)
        })?;
    let mut model = BetaCrownModel::Sequential(Box::new(seq_net));
    apply_heuristic_sound_modes(
        &mut model,
        allow_heuristic_logsoftmax,
        allow_heuristic_softmax,
        json,
    );
    Ok(model)
}

#[cfg(feature = "mip")]
fn phase_telemetry_enabled_from_value(value: Option<&std::ffi::OsStr>) -> bool {
    value == Some(std::ffi::OsStr::new("1"))
}

#[cfg(feature = "mip")]
fn phase_telemetry_enabled() -> bool {
    phase_telemetry_enabled_from_value(std::env::var_os("NY_PHASE_TELEMETRY").as_deref())
}

#[cfg(feature = "mip")]
fn safenlp_direct_first_marker_if(enabled: bool, event: &str) -> Option<String> {
    enabled.then(|| {
        format!(
            "NY_MIP_SAFENLP_DIRECT_FIRST_V1 event={event} route=direct-mip-first \
             shared_prefix=required internal_pgd=false warm_start=false \
             deadline=caller-absolute"
        )
    })
}

#[cfg(feature = "mip")]
fn safenlp_direct_first_decline_marker_if(
    enabled: bool,
    reason: &SafeNlpDirectMipFirstDecline,
) -> Option<String> {
    if !enabled || matches!(reason, SafeNlpDirectMipFirstDecline::NotRequested) {
        return None;
    }
    let (stage, code) = reason.telemetry();
    Some(format!(
        "NY_MIP_SAFENLP_DIRECT_FIRST_V1 event=route-decline \
         stage={stage} reason={code} historical_fallback=bab"
    ))
}

#[cfg(feature = "mip")]
fn emit_safenlp_direct_first_decline(reason: &SafeNlpDirectMipFirstDecline) {
    if matches!(reason, SafeNlpDirectMipFirstDecline::NotRequested) {
        return;
    }
    let (stage, code) = reason.telemetry();
    tracing::info!(
        stage,
        reason = code,
        detail = ?reason,
        "SafeNLP direct-MIP-first route declined before start; preserving historical BaB"
    );
    if let Some(marker) = safenlp_direct_first_decline_marker_if(phase_telemetry_enabled(), reason)
    {
        eprintln!("{marker}");
    }
}

/// Apply the exact policy chain used by the production BaB -> MIP dispatch.
///
/// The SafeNLP treatment is inserted before static graph ineligibility so an
/// already-armed AUTO/MIP policy can retain its slice for the later sequential
/// reload. It never re-arms explicit `bab` or any otherwise-disarmed ledger.
fn apply_bab_mip_budget_policy(
    ledger: PhaseBudgetLedger,
    complete_verifier: CompleteVerifierArg,
    graph_mip_statically_ineligible: bool,
    safenlp_shared_prefix_budget_repair: bool,
) -> PhaseBudgetLedger {
    ledger
        .with_mip_escalation_allowed(!matches!(complete_verifier, CompleteVerifierArg::Bab))
        .with_safenlp_shared_prefix_budget_repair(safenlp_shared_prefix_budget_repair)
        .with_static_mip_ineligibility(graph_mip_statically_ineligible)
}

/// Graph-level form of [`graph_mip_statically_ineligible`]. `false` in a non-mip
/// build (the reservation is never armed there, so the answer cannot matter).
pub(super) fn graph_mip_layer_set_statically_unsupported(
    graph: &ny_propagate::GraphNetwork,
) -> bool {
    #[cfg(feature = "mip")]
    {
        !graph_mip_layer_set_supported(graph)
    }
    #[cfg(not(feature = "mip"))]
    {
        let _ = graph;
        false
    }
}

/// #deadlane — the purely STATIC half of [`graph_unstable_relu_count`]'s
/// eligibility test: is every node in the graph inside the Graph-MIP encoder's
/// layer set?
///
/// `graph_unstable_relu_count` returns `None` (fail-closed, "NOT eligible") for
/// any layer outside `{Linear, Conv2d, Flatten, Reshape, BatchNorm, Add, ReLU}`
/// REGARDLESS of the bounds it is handed, so a `false` here means the whole-net
/// Graph-MIP escalation can NEVER fire for this model — and the BaB deadline must
/// not reserve a slice for it. Measured on vit_2023 (`Shape`/`Gather`/`Slice`/
/// `Concat`/`Unsqueeze`/`ConstantOfShape`/`ReduceMean`) and yolo_2023 (`Pad`):
/// every row logs "Graph-MIP: NOT eligible (unsupported layer or a ReLU without a
/// pre-activation box)" while the reservation still cost BaB 23 s of its 95 s
/// internal tier.
///
/// Keep the match arms in lockstep with `graph_unstable_relu_count` below —
/// this is deliberately the SAME set, minus the bounds-dependent ReLU box
/// lookup, so it can only be TRUE where that function could also succeed.
#[cfg(feature = "mip")]
pub(super) fn graph_mip_layer_set_supported(graph: &ny_propagate::GraphNetwork) -> bool {
    use ny_propagate::Layer;
    let Ok(exec) = graph.exec_order() else {
        // No execution order => the encoder could not walk the net either.
        return false;
    };
    exec.iter().all(|name| {
        graph.node(name).is_some_and(|node| {
            matches!(
                node.layer(),
                Layer::Linear(_)
                    | Layer::Conv2d(_)
                    | Layer::Flatten(_)
                    | Layer::Reshape(_)
                    | Layer::BatchNorm(_)
                    | Layer::Add(_)
                    | Layer::ReLU(_)
            )
        })
    })
}

/// Count the UNSTABLE ReLU pre-activation neurons (`l < 0 < u`) under the
/// supplied bounds — the big-M binary count a Graph-MIP encode would carry.
/// `None` (fail-closed) for any layer outside the graph encoder's set or a
/// ReLU without a pre-activation box (the encoder would bail). Shared by the
/// eligibility gate and its NOT-eligible visibility line.
#[cfg(feature = "mip")]
pub(super) fn graph_unstable_relu_count(
    graph: &ny_propagate::GraphNetwork,
    node_bounds: &std::collections::HashMap<String, Vec<ny_core::Bound>>,
) -> Option<usize> {
    use ny_propagate::Layer;
    let exec = graph.exec_order().ok()?;
    let mut unstable: usize = 0;
    for name in exec {
        let node = graph.node(name)?;
        match node.layer() {
            Layer::Linear(_)
            | Layer::Conv2d(_)
            | Layer::Flatten(_)
            | Layer::Reshape(_)
            | Layer::BatchNorm(_)
            | Layer::Add(_) => {}
            Layer::ReLU(_) => {
                // Pre-activation box lookup mirrors the encoder: the ReLU's own
                // entry first, else its input node's (affine producer's) box.
                let pre = node_bounds.get(name).or_else(|| {
                    node.inputs()
                        .first()
                        .and_then(|input| node_bounds.get(input))
                })?;
                unstable += pre
                    .iter()
                    .filter(|b| b.lower() < 0.0 && b.upper() > 0.0)
                    .count();
            }
            _ => return None,
        }
    }
    Some(unstable)
}

/// Pure auto-escalation policy decision.
///
/// Returns `true` iff we should escalate the BaB result to the MIP complete
/// verifier. Escalation is gated on three conditions, all of which preserve
/// soundness:
///   1. the effective policy is "escalate" — either an explicit
///      `--complete-verifier mip`, or the default `auto` (NOT explicit `bab`);
///   2. BaB did not decide the property (only `Timeout`/`Unknown` escalate —
///      a `Verified`/`Violated`/`PotentialViolation` BaB verdict is kept);
///   3. the network is MIP-encodable within the size cap.
///
/// The certificate-gated Big-M MIP path is a sound complete verifier, so escalating can
/// only turn `unknown`/`timeout` into a decided verdict, never flip a decided
/// one. Condition (2) guarantees we never discard a BaB decision.
#[cfg(feature = "mip")]
pub(super) fn should_auto_escalate_to_mip(
    complete_verifier: CompleteVerifierArg,
    bab_status: &BabVerificationStatus,
    network_is_mip_encodable: bool,
) -> bool {
    let policy_allows = matches!(
        complete_verifier,
        CompleteVerifierArg::Mip | CompleteVerifierArg::Auto
    );
    let bab_inconclusive = matches!(
        bab_status,
        BabVerificationStatus::Timeout | BabVerificationStatus::Unknown { .. }
    );
    policy_allows && bab_inconclusive && network_is_mip_encodable
}

#[cfg(feature = "mip")]
fn affine_root_farkas_policy_allows(
    complete_verifier: CompleteVerifierArg,
    mip_solver: MipSolverArg,
) -> bool {
    matches!(
        (complete_verifier, mip_solver),
        (
            CompleteVerifierArg::Auto | CompleteVerifierArg::Mip,
            MipSolverArg::AY
        )
    )
}

/// Shared verification dispatch inputs assembled by the CLI handler.
// `model_path`, `onnx_load_config`, and `allow_heuristic_*` are read only by the
// `#[cfg(feature = "mip")]` escalation/reload path; allow them as dead code when
// the mip feature is off so the non-mip build stays warning-free.
#[cfg_attr(not(feature = "mip"), allow(dead_code))]
pub(super) struct DispatchContext<'a> {
    pub(super) model_path: &'a Path,
    pub(super) onnx_load_config: &'a OnnxLoadConfig,
    pub(super) model_net: &'a mut BetaCrownModel,
    pub(super) input: &'a BoundedTensor,
    pub(super) config: &'a BetaCrownConfig,
    pub(super) effective_treatment: &'a EffectiveTreatmentProjection,
    pub(super) vnnlib_spec: Option<&'a VnnLibSpec>,
    pub(super) property: &'a Option<PathBuf>,
    pub(super) epsilon: f32,
    pub(super) effective_threshold: f32,
    /// `Y_i >= c` unsafe constraint: the verifier proves `output < c` rather
    /// than `output > c`. Drives the direction words in `output_result`.
    pub(super) verify_upper: bool,
    pub(super) output_dim: usize,
    pub(super) const_output_idx: Option<usize>,
    pub(super) has_relational: bool,
    pub(super) use_relu_split: bool,
    pub(super) gpu_bab: bool,
    pub(super) run_upfront_pgd: bool,
    /// Compat-free `after` is owned by the standard Sequential engine. Its
    /// caller must pass a horizon that has not already removed the same PGD
    /// fraction.
    pub(super) engine_owns_deferred_pgd: bool,
    /// Compat-free `after` is owned by VNN-COMP's outer wrapper. The internal
    /// verifier must neither execute PGD nor reserve a second copy of its
    /// fraction.
    pub(super) outer_wrapper_owns_deferred_pgd: bool,
    /// Typed category/budget intent from the in-process VNN-COMP router.
    /// Dispatch must still satisfy every structural/deadline/shared-prefix
    /// prerequisite before this may bypass BaB.
    pub(super) safenlp_direct_mip_first: bool,
    /// Typed, exact-category intent for the default-dark imgSz32 cGAN
    /// input-leaf oracle. Model/property/row authentication remains local to
    /// the attachment seam.
    pub(super) cgan_input_leaf_route: Option<super::CganInputLeafRoute>,
    pub(super) gemm_engine: Option<&'a dyn GemmEngine>,
    /// #attack-steering-unquarantine: live accelerator for falsification
    /// STEERING only (batched / exact-VJP PGD lanes). Never handed to
    /// bound/precheck/BaB work — those stay on the quarantined proof adapter.
    /// Arms in the background (#wallhugger-arming-cost): attack lanes take it
    /// non-blockingly at their call sites and run un-steered until ready.
    pub(super) attack_engine_source: super::attack_arming::AttackEngineSource<'a>,
    pub(super) compute_device: Option<Arc<ComputeDevice>>,
    pub(super) allow_heuristic_logsoftmax: bool,
    pub(super) allow_heuristic_softmax: bool,
    pub(super) input_split_metrics_jsonl: Option<&'a Path>,
    pub(super) domain_batch_metrics_jsonl: Option<&'a Path>,
    /// Start of the complete verification attempt, including exact-CNF work.
    pub(super) verification_start: Instant,
    /// One authoritative wall-clock deadline shared with pre-dispatch routes.
    /// `None` means unbounded.
    pub(super) overall_deadline: Option<Instant>,
    /// VNN-COMP's independently resolved outer post-BaB attack route.
    /// `None` preserves the historical interactive small-budget reserve.
    pub(super) post_bab_wrapper_attack_enabled: Option<bool>,
    pub(super) json: bool,
    /// Loader auto-peeled a terminal Sigmoid (#cgan-sigmoid-peel): witness
    /// outputs are pre-sigmoid logits and must be mapped y = sigmoid(z) at
    /// emission so declared Y matches the ORIGINAL graph.
    pub(super) sigmoid_peeled: bool,
    /// Proof-carrying / certificate-emission options (default on; off in
    /// competition mode). Consumed by [`super::cert_adapter`] at the
    /// verdict-emission chokepoints below.
    pub(super) proof_opts: &'a super::ProofOpts,
}

/// Whether the shared BaB dispatch will select the relational verifier rather
/// than `verify_standard`. Keep admission checks at the CLI boundary on this
/// exact predicate: only `verify_standard` owns the engine's internal deferred
/// PGD fallback.
pub(super) fn routes_to_relational_verifier(
    vnnlib_spec: Option<&VnnLibSpec>,
    has_relational: bool,
) -> bool {
    vnnlib_spec.is_some_and(|vnnlib| {
        vnnlib.output_constraints.len() > 1
            || has_relational
            || vnnlib.has_boxed_clause_disjunction()
    })
}

fn verifier_config_for_deferred_pgd_owner(
    config: &BetaCrownConfig,
    outer_wrapper_owns_deferred_pgd: bool,
) -> BetaCrownConfig {
    let mut config = config.clone();
    if outer_wrapper_owns_deferred_pgd {
        config.enable_pgd_attack = false;
        config.phase_budget.post_bab_pgd_fraction = 0.0;
    }
    config
}

fn ledger_policy_for_deferred_pgd_owner(
    config: &BetaCrownConfig,
    engine_owns_deferred_pgd: bool,
    outer_wrapper_owns_deferred_pgd: bool,
) -> ny_propagate::PhaseBudgetConfig {
    let mut policy = config.phase_budget.clone();
    if engine_owns_deferred_pgd || outer_wrapper_owns_deferred_pgd {
        policy.post_bab_pgd_fraction = 0.0;
    }
    policy
}

fn potential_violation_confirmation_deadline(
    ledger: &PhaseBudgetLedger,
    outer_wrapper_owns_deferred_pgd: bool,
) -> Option<Instant> {
    if outer_wrapper_owns_deferred_pgd {
        ledger.bab_deadline()
    } else {
        ledger.overall_deadline()
    }
}

/// Normalize elapsed time and enforce the one authoritative wall deadline at
/// every verdict-emission seam.
///
/// Individual engines receive the same deadline and normally stop themselves,
/// but some foundational passes (ONNX reload, IBP, structural recognizers, MIP
/// preprocessing) are synchronous.  They may finish just after their budget.
/// A result computed after the deadline is still mathematically valid, but it
/// is not admissible for the bounded verification attempt, so report Timeout
/// instead of publishing a late Verified/Violated verdict.
pub(super) fn gate_result_at_deadline(
    result: BetaCrownResult,
    verification_start: Instant,
    deadline: Option<Instant>,
) -> BetaCrownResult {
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return BetaCrownResult {
            result: BabVerificationStatus::Timeout,
            domains_explored: result.domains_explored,
            domains_verified: result.domains_verified,
            cuts_generated: result.cuts_generated,
            max_depth_reached: result.max_depth_reached,
            time_elapsed: verification_start.elapsed(),
            output_bounds: None,
        };
    }
    normalize_result_wall_time(result, verification_start)
}

/// Recover a finite context deadline for direct internal callers that provide
/// a bounded config but omit the normally-authoritative handler deadline.
fn ledger_deadline(ctx: &DispatchContext<'_>) -> Option<Instant> {
    ctx.overall_deadline.or_else(|| {
        (!ctx.config.timeout.is_zero())
            .then(|| ctx.verification_start.checked_add(ctx.config.timeout))
            .flatten()
    })
}

/// Translate a bubbled verifier deadline into the ordinary typed timeout
/// result consumed by the rest of the CLI dispatch.
///
/// The propagation crates expose deadline expiry as
/// `NyError::DeadlineExceeded`. `anyhow` context may sit above that source, so
/// inspect the complete error chain instead of matching the rendered string.
/// The configured authority must actually be expired: an unbounded caller, or
/// a caller whose outer/phase deadline is still live, must not reinterpret a
/// recoverable node-local deadline as exhaustion of its verification budget.
fn timeout_result_from_deadline_error(
    error: &anyhow::Error,
    verification_start: Instant,
    deadline: Option<Instant>,
) -> Option<BetaCrownResult> {
    let has_typed_deadline = error.chain().any(|cause| {
        cause
            .downcast_ref::<ny_core::NyError>()
            .is_some_and(ny_core::NyError::is_deadline_exceeded)
    });
    let authority_expired = deadline.is_some_and(|deadline| Instant::now() >= deadline);
    (authority_expired && has_typed_deadline).then(|| BetaCrownResult {
        result: BabVerificationStatus::Timeout,
        domains_explored: 0,
        domains_verified: 0,
        cuts_generated: 0,
        max_depth_reached: 0,
        time_elapsed: verification_start.elapsed(),
        output_bounds: None,
    })
}

/// Run the sequential MIP-only path, including the PGD warm-start precheck.
pub(super) fn run_mip_only(ctx: &DispatchContext<'_>, mip_solver: MipSolverArg) -> Result<()> {
    run_mip_only_with_seed_policy(
        ctx,
        &*ctx.model_net,
        mip_solver,
        MipOnlySeedPolicy::HistoricalConfigured,
    )
}

fn run_mip_only_with_seed_policy(
    ctx: &DispatchContext<'_>,
    model_net: &BetaCrownModel,
    mip_solver: MipSolverArg,
    seed_policy: MipOnlySeedPolicy,
) -> Result<()> {
    if ctx.overall_deadline.is_none() {
        anyhow::bail!("unbounded MIP verification is unsupported; pass a positive --timeout");
    }
    let mip_ledger = PhaseBudgetLedger::from_deadline(
        ctx.overall_deadline,
        ledger_policy_for_deferred_pgd_owner(
            ctx.config,
            ctx.engine_owns_deferred_pgd,
            ctx.outer_wrapper_owns_deferred_pgd,
        ),
    )
    .with_post_bab_wrapper_attack(ctx.post_bab_wrapper_attack_enabled);
    let internal_authority_deadline = mip_ledger.overall_deadline();

    let warm_start_candidate = if seed_policy.run_internal_pgd(ctx.run_upfront_pgd) {
        match model_net {
            BetaCrownModel::Sequential(network) => {
                if let Some(vnnlib) = ctx.vnnlib_spec {
                    // Cap the PGD precheck at the upfront-PGD phase budget
                    // (default 20%), NOT the overall deadline: PGD is a pure
                    // falsifier and can never prove UNSAT, so letting it run
                    // to the overall deadline starved the exact MIP solver of
                    // its entire budget on large UNSAT instances (measured:
                    // sat_relu unsat_v90_c111 spent all 95s in the 1000-restart
                    // sampling attack and MIP never ran). Mirrors the BaB
                    // path (verify/graph.rs), which already uses
                    // upfront_pgd_deadline(). SAT instances still short-circuit
                    // long before the cap when PGD finds a counterexample.
                    let deadline = mip_ledger.upfront_pgd_deadline();
                    let early_pgd = try_pgd_before_mip(
                        network,
                        ctx.input,
                        vnnlib,
                        ctx.config.pgd_restarts,
                        ctx.config.pgd_steps,
                        ctx.config.pgd_initialization,
                        ctx.config.pgd_osi_steps,
                        deadline,
                        ctx.config.pgd_restart_when_stuck,
                        ctx.gemm_engine,
                        ctx.json,
                    )?;
                    if let Some((counterexample, output)) = early_pgd.confirmed_counterexample {
                        let result = gate_result_at_deadline(
                            BetaCrownResult {
                                result: BabVerificationStatus::Violated {
                                    counterexample: counterexample.iter().copied().collect(),
                                    output: output.iter().copied().collect(),
                                },
                                domains_explored: 0,
                                domains_verified: 0,
                                cuts_generated: 0,
                                max_depth_reached: 0,
                                time_elapsed: ctx.verification_start.elapsed(),
                                output_bounds: None,
                            },
                            ctx.verification_start,
                            internal_authority_deadline,
                        );
                        output_result(
                            &result,
                            ctx.property,
                            Some(ctx.model_path),
                            ctx.epsilon,
                            ctx.effective_threshold,
                            ctx.verify_upper,
                            ctx.json,
                            ctx.sigmoid_peeled,
                            ctx.effective_treatment,
                        )?;
                        return Ok(());
                    }
                    early_pgd.warm_start_candidate
                } else {
                    None
                }
            }
            BetaCrownModel::Graph(_) => None,
        }
    } else {
        None
    };

    let Some(internal_authority_deadline) = internal_authority_deadline else {
        unreachable!("the unbounded MIP case is rejected before the PGD precheck");
    };
    if Instant::now() >= internal_authority_deadline {
        let result = BetaCrownResult {
            result: BabVerificationStatus::Timeout,
            domains_explored: 0,
            domains_verified: 0,
            cuts_generated: 0,
            max_depth_reached: 0,
            time_elapsed: ctx.verification_start.elapsed(),
            output_bounds: None,
        };
        output_result(
            &result,
            ctx.property,
            Some(ctx.model_path),
            ctx.epsilon,
            ctx.effective_threshold,
            ctx.verify_upper,
            ctx.json,
            ctx.sigmoid_peeled,
            ctx.effective_treatment,
        )?;
        return Ok(());
    }

    // Every solver arg routes through the same encoder and witness/proof
    // gates. The typed direct-first ingress additionally requires the AY
    // shared-binary-prefix session; an internal admission decline is an
    // error/unknown under this clock and may not launch another MIP route.
    match seed_policy {
        MipOnlySeedPolicy::HistoricalConfigured => mip_highs::verify_with_mip(
            model_net,
            ctx.input,
            ctx.vnnlib_spec,
            ctx.property.as_deref(),
            Some(ctx.model_path),
            ctx.epsilon,
            ctx.effective_threshold,
            internal_authority_deadline,
            warm_start_candidate.as_ref(),
            mip_solver,
            ctx.verification_start,
            ctx.effective_treatment,
            ctx.json,
        ),
        #[cfg(feature = "mip")]
        MipOnlySeedPolicy::ColdNoWarmStart => {
            debug_assert!(warm_start_candidate.is_none());
            mip_highs::verify_with_mip_requiring_safenlp_shared_prefix(
                model_net,
                ctx.input,
                ctx.vnnlib_spec,
                ctx.property.as_deref(),
                Some(ctx.model_path),
                ctx.epsilon,
                ctx.effective_threshold,
                internal_authority_deadline,
                None,
                mip_solver,
                ctx.verification_start,
                ctx.effective_treatment,
                ctx.json,
            )
        }
    }
}

#[cfg(feature = "mip")]
fn run_safenlp_direct_mip_first(
    ctx: &DispatchContext<'_>,
    reloaded_model: &BetaCrownModel,
    mip_solver: MipSolverArg,
    plan: SafeNlpDirectMipFirstPlan,
    source: SafeNlpDirectMipModelSource,
) -> Result<()> {
    debug_assert_eq!(
        ctx.overall_deadline,
        Some(plan.deadline),
        "direct-first must consume the caller's immutable absolute deadline"
    );
    debug_assert_eq!(plan.seed_policy, MipOnlySeedPolicy::ColdNoWarmStart);
    debug_assert!(!plan.seed_policy.permits_warm_start());
    debug_assert_eq!(plan.solver_route, MipOnlySolverRoute::SharedBinaryPrefix);
    if let Some(marker) = safenlp_direct_first_marker_if(phase_telemetry_enabled(), "route-start") {
        eprintln!("{marker}");
    }
    tracing::info!(
        hidden_dim = plan.hidden_dim,
        source_model = source.telemetry_name(),
        reloaded_model = "sequential",
        internal_pgd = false,
        warm_start = false,
        shared_prefix = "required",
        "SafeNLP direct-MIP-first route admitted"
    );
    let result = run_mip_only_with_seed_policy(ctx, reloaded_model, mip_solver, plan.seed_policy);
    if let Some(marker) = safenlp_direct_first_marker_if(phase_telemetry_enabled(), "route-return")
    {
        eprintln!("{marker}");
    }
    result
}

/// Run the BaB path, then optionally fall back to MIP if the result is inconclusive.
pub(super) fn run_bab_with_fallback(
    ctx: &mut DispatchContext<'_>,
    complete_verifier: CompleteVerifierArg,
    mip_solver: MipSolverArg,
) -> Result<()> {
    // The MIP-escalation policy below consumes these; in a non-mip build the
    // escalation block is compiled out, so acknowledge them to stay warning-free.
    #[cfg(not(feature = "mip"))]
    let _ = mip_solver;
    // #star-measure: verdict-neutral OBSERVATION of the exact star split search on the
    // real scored row. Inert unless an operator arms `NY_STAR_DARK_SECONDS`; it returns
    // `()`, so no branch below can read it and no verdict can depend on it. It sits at
    // the top of `run_bab_with_fallback` so the search gets a whole budget rather than
    // whatever the shipped pipeline leaves behind.
    //
    // This is NOT ahead of every lane: `run_bab_with_fallback` is one dispatch route
    // among several, and rows the upfront attack settles never reach it at all
    // (measured — acasxu 1_4/prop_2 returns `sat` in 0.25s with the probe armed and the
    // probe never runs). So the probe observes BaB-bound rows only, which is exactly the
    // population the acasxu residue lives in.
    #[cfg(feature = "mip")]
    if let (BetaCrownModel::Sequential(network), Some(property)) =
        (&*ctx.model_net, ctx.property.as_deref())
    {
        super::star_candidate::run_dark_star_probe(network, property);
    }
    // One ledger owns the overall wall-clock deadline in every build.  MIP
    // builds additionally consume its escalation budget, while all builds use
    // the same deadline to bound optional post-verdict certificate emission.
    // #deadlane: give BaB the Graph-MIP slice back when the escalation is
    // statically impossible on this net (see `with_static_mip_ineligibility`).
    let bab_ledger = apply_bab_mip_budget_policy(
        PhaseBudgetLedger::from_deadline(
            ledger_deadline(ctx),
            ledger_policy_for_deferred_pgd_owner(
                ctx.config,
                ctx.engine_owns_deferred_pgd,
                ctx.outer_wrapper_owns_deferred_pgd,
            ),
        )
        .with_post_bab_wrapper_attack(ctx.post_bab_wrapper_attack_enabled),
        complete_verifier,
        graph_mip_statically_ineligible(&*ctx.model_net),
        safenlp_shared_prefix_budget_repair_enabled(),
    );
    bab_ledger.emit_telemetry("bab-dispatch");
    let internal_authority_deadline = bab_ledger.overall_deadline();
    if let Some(vnnlib) = ctx.vnnlib_spec {
        let ibp_result = match &*ctx.model_net {
            BetaCrownModel::Sequential(network) => network.propagate_ibp(ctx.input),
            BetaCrownModel::Graph(graph) => graph.propagate_ibp(ctx.input),
        };
        if let Ok(ibp_bounds) = ibp_result {
            let ibp_lower = ibp_bounds.lower().as_slice().unwrap_or(&[]);
            let ibp_upper = ibp_bounds.upper().as_slice().unwrap_or(&[]);
            if ibp_check_vnnlib_safe(ibp_lower, ibp_upper, vnnlib) {
                if !ctx.json {
                    info!("IBP fast-path: all constraints verified by IBP alone");
                }
                let result = BetaCrownResult {
                    result: BabVerificationStatus::Verified,
                    domains_explored: 0,
                    time_elapsed: std::time::Duration::from_millis(0),
                    max_depth_reached: 0,
                    output_bounds: Some(ibp_bounds),
                    cuts_generated: 0,
                    domains_verified: 0,
                };
                // Verdict FIRST, certificate second: emission is post-verdict,
                // optional, and budget-bounded — but any future pathology there
                // must never again delay (or, on a panic, mask) an
                // already-decided verdict. Cert logs go to stderr, so JSON
                // stdout stays clean either way.
                let result = gate_result_at_deadline(
                    result,
                    ctx.verification_start,
                    internal_authority_deadline,
                );
                output_result(
                    &result,
                    ctx.property,
                    Some(ctx.model_path),
                    ctx.epsilon,
                    ctx.effective_threshold,
                    ctx.verify_upper,
                    ctx.json,
                    ctx.sigmoid_peeled,
                    ctx.effective_treatment,
                )?;
                super::cert_adapter::maybe_emit_certificate(
                    ctx,
                    &result,
                    internal_authority_deadline,
                );
                return Ok(());
            }
        }
    }

    // Exact affine INVPROP closure: after preserving every cheap IBP win, try
    // to prove that the ONE conjunctive candidate-violation region is empty for
    // a pure affine/fixed-phase sequential model. This optional lane is present
    // only in the competition MIP tier and has runtime authority only when no
    // external certificate was requested. Every refusal falls through to the
    // historical direct-MIP/cell-enum/BaB dispatch below.
    #[cfg(feature = "mip")]
    if affine_root_farkas_policy_allows(complete_verifier, mip_solver) {
        if let Some(vnnlib) = ctx.vnnlib_spec {
            if let super::affine_invprop::AffineRootFarkasAttempt::Proved(certificate) =
                super::affine_invprop::try_affine_root_farkas(
                    ctx.model_path,
                    ctx.onnx_load_config,
                    ctx.property.as_deref(),
                    ctx.input,
                    vnnlib,
                    ctx.config.verification_artifact_authority,
                    internal_authority_deadline,
                    ctx.sigmoid_peeled,
                )
            {
                info!(
                    ay_farkas_multipliers = certificate.ay_farkas_multipliers,
                    ny_cert_farkas_replays = certificate.ny_cert_farkas_replays,
                    "exact affine root-Farkas closure proved the candidate-violation region empty"
                );
                let result = gate_result_at_deadline(
                    BetaCrownResult {
                        result: BabVerificationStatus::Verified,
                        domains_explored: 0,
                        domains_verified: 0,
                        cuts_generated: 0,
                        max_depth_reached: 0,
                        time_elapsed: std::time::Duration::ZERO,
                        output_bounds: None,
                    },
                    ctx.verification_start,
                    internal_authority_deadline,
                );
                output_result(
                    &result,
                    ctx.property,
                    Some(ctx.model_path),
                    ctx.epsilon,
                    ctx.effective_threshold,
                    ctx.verify_upper,
                    ctx.json,
                    ctx.sigmoid_peeled,
                    ctx.effective_treatment,
                )?;
                return Ok(());
            }
        }
    }

    // Preserve the existing authoritative IBP win before admitting the
    // category-wide direct-first experiment. The BaB model may intentionally
    // be Graph-shaped even when the exact ONNX has a canonical sequential
    // Linear-ReLU-Linear form, so an armed row uses the same fresh sequential
    // reload seam as historical post-BaB MIP escalation. Reload and admission
    // consume the original absolute deadline; every pre-route decline falls
    // through without emitting `route-start`.
    #[cfg(feature = "mip")]
    if ctx.safenlp_direct_mip_first {
        let source = SafeNlpDirectMipModelSource::from_model(&*ctx.model_net);
        let model_path = ctx.model_path;
        let onnx_load_config = ctx.onnx_load_config;
        let allow_heuristic_logsoftmax = ctx.allow_heuristic_logsoftmax;
        let allow_heuristic_softmax = ctx.allow_heuristic_softmax;
        let json = ctx.json;
        let attempt = safenlp_direct_mip_first_attempt_with_reload(
            true,
            safenlp_shared_prefix_budget_repair_enabled(),
            complete_verifier,
            mip_solver,
            source,
            ctx.input,
            ctx.vnnlib_spec,
            internal_authority_deadline,
            Instant::now(),
            || {
                reload_safenlp_direct_mip_model(
                    model_path,
                    onnx_load_config,
                    allow_heuristic_logsoftmax,
                    allow_heuristic_softmax,
                    json,
                )
            },
            Instant::now,
            |reloaded_model, plan, source| {
                run_safenlp_direct_mip_first(ctx, &reloaded_model, mip_solver, plan, source)
            },
        );
        match attempt {
            SafeNlpDirectMipFirstAttempt::Declined(reason) => {
                emit_safenlp_direct_first_decline(&reason);
            }
            SafeNlpDirectMipFirstAttempt::Executed(result) => return result,
        }
    }

    // Cell-enumeration driver (#cctsdb Phase C): piecewise-constant models
    // whose free inputs are Trunc-gated (cctsdb_yolo_2023) are decided by
    // enumerating integer cells with a sound f64 per-cell forward — BEFORE
    // BaB, which cannot decide them (the mask hull keeps Y unbounded).
    // Structurally gated + fail-closed: non-qualifying specs fall through
    // with zero behavior change. Disable with NY_NO_CELL_ENUM=1.
    if let Some(vnnlib) = ctx.vnnlib_spec {
        if let Some(result) = super::cell_enum::try_cell_enumeration(
            &*ctx.model_net,
            ctx.input.shape(),
            vnnlib,
            internal_authority_deadline,
        ) {
            // A `Violated` result carries a concrete, in-process-confirmed
            // witness; the vnncomp harness re-confirms it against the
            // ONNX-Runtime trusted oracle before any `sat` is scored.
            let result = gate_result_at_deadline(
                result,
                ctx.verification_start,
                internal_authority_deadline,
            );
            output_result(
                &result,
                ctx.property,
                Some(ctx.model_path),
                ctx.epsilon,
                ctx.effective_threshold,
                ctx.verify_upper,
                ctx.json,
                ctx.sigmoid_peeled,
                ctx.effective_treatment,
            )?;
            return Ok(());
        }
    }

    // Normalized-power fractional-head driver (nn4sys pensieve `*_parallel`):
    // `Y = Sub` of two `Linear(Div(Pow(relu,k), ReduceSum(Pow)))` heads. The
    // generic Div relaxation is hopeless here (root gap ~120 units); the
    // structural driver bounds each head's linear-fractional tail EXACTLY
    // (threshold-vertex enumeration, outward f64 rounding) over prefix-CROWN
    // logit bounds — BEFORE standard BaB. Structurally gated + fail-open:
    // non-qualifying specs fall through with zero behavior change. Any sat
    // witness is re-confirmed by the vnncomp ORT gate downstream. Disable
    // with NY_NO_FRAC_HEAD=1.
    if let Some(vnnlib) = ctx.vnnlib_spec {
        if let Some(result) = super::frac_head::try_frac_head_verification(
            &*ctx.model_net,
            ctx.input.shape(),
            vnnlib,
            internal_authority_deadline,
        ) {
            let result = gate_result_at_deadline(
                result,
                ctx.verification_start,
                internal_authority_deadline,
            );
            output_result(
                &result,
                ctx.property,
                Some(ctx.model_path),
                ctx.epsilon,
                ctx.effective_threshold,
                ctx.verify_upper,
                ctx.json,
                ctx.sigmoid_peeled,
                ctx.effective_treatment,
            )?;
            return Ok(());
        }
    }

    let execution_config =
        verifier_config_for_deferred_pgd_owner(ctx.config, ctx.outer_wrapper_owns_deferred_pgd);
    let mut verifier = match ctx.compute_device.clone() {
        Some(device) => BetaCrownVerifier::new_with_engine(execution_config.clone(), device),
        None => BetaCrownVerifier::new(execution_config.clone()),
    };
    if let Some(path) = ctx.input_split_metrics_jsonl {
        verifier =
            verifier.with_input_split_metrics_sink(JsonlInputSplitMetricsSink::create(path)?);
    }
    if let Some(path) = ctx.domain_batch_metrics_jsonl {
        verifier = verifier
            .with_graph_domain_batch_metrics_sink(JsonlGraphDomainBatchMetricsSink::create(path)?);
    }

    if let BetaCrownModel::Graph(graph) = &mut *ctx.model_net {
        graph.set_use_patches_mode(execution_config.use_patches());
    }

    #[cfg(feature = "mip")]
    let mut cgan_input_leaf_attachment: Option<
        super::cgan_input_leaf::CganInputLeafAttachment,
    > = None;

    // Graph-MIP LEAF oracle (increment 6, `docs/GRAPH_MIP_LEAF_SOLVER.md`):
    // the graph ReLU-split BaB decides stuck deep subdomains exactly (split
    // premises pinned, certified-UNSAT admission) instead of requeueing them.
    // DEFAULT-ON (2026-07-21, sound + time-sliced); `NY_GRAPH_MIP_LEAF=0`
    // detaches the oracle ⇒ every BaB path byte-identical (the old default).
    #[cfg(feature = "mip")]
    {
        let mut leaf_oracles: Vec<
            Arc<dyn ny_propagate::beta_crown::graph_mip_leaf::GraphMipLeafOracle>,
        > = Vec::new();
        if let (
            Some(route),
            BetaCrownModel::Graph(graph),
            Some(property),
            Some(vnnlib),
            Some(authority_deadline),
        ) = (
            ctx.cgan_input_leaf_route,
            &*ctx.model_net,
            ctx.property.as_deref(),
            ctx.vnnlib_spec,
            bab_ledger.bab_deadline(),
        ) {
            if let Some(attachment) = super::cgan_input_leaf::maybe_cgan_input_leaf_oracle(
                route,
                authority_deadline,
                ctx.model_path,
                property,
                ctx.onnx_load_config,
                graph,
                ctx.input,
                vnnlib,
            ) {
                crate::flight::note(
                    "cgan_input_leaf_attachment",
                    crate::flight::FlightStatus::Ran,
                    Some(
                        "exact category/model/property/profile/root authentication completed"
                            .to_string(),
                    ),
                );
                leaf_oracles.push(attachment.oracle());
                cgan_input_leaf_attachment = Some(attachment);
            } else {
                crate::flight::note(
                    "cgan_input_leaf_attachment",
                    crate::flight::FlightStatus::Skipped,
                    Some(
                        "typed route reached dispatch but exact authentication declined"
                            .to_string(),
                    ),
                );
            }
        }
        if let Some(oracle) = super::graph_mip_leaf::maybe_graph_mip_leaf_oracle(
            mip_solver.mip_backend(),
            ctx.vnnlib_spec,
        ) {
            leaf_oracles.push(oracle);
        }
        verifier = match leaf_oracles.len() {
            0 => verifier,
            1 => verifier.with_graph_mip_leaf_oracle(leaf_oracles.pop().unwrap()),
            _ => verifier.with_graph_mip_leaf_oracle(Arc::new(
                crate::commands::coupled_delta::CompositeLeafOracle::new(leaf_oracles),
            )),
        };
    }

    // Single-clause per-clause-box disjunctions (nn4sys mscn `_dual`
    // cardinality_1_1: one `(and <input box> <band constraint>)` disjunct)
    // parse to ONE output constraint, so the plain `len() > 1` gate used to
    // send them to `verify_standard` — whose global-box f32 BaB can never
    // decide a ±1e-5 band. Route them through the relational dispatch, which
    // forwards per-clause-box shapes to the box-refinement screen (+ sound
    // f64 leaf escalation). Verdict semantics are identical for one clause.
    let has_multi_constraints = routes_to_relational_verifier(ctx.vnnlib_spec, ctx.has_relational);

    let verification = if let (true, Some(vnnlib)) = (has_multi_constraints, ctx.vnnlib_spec) {
        verify_relational_constraints_with_ledger(
            &*ctx.model_net,
            ctx.input,
            vnnlib,
            &execution_config,
            &verifier,
            ctx.use_relu_split,
            ctx.gpu_bab,
            ctx.run_upfront_pgd,
            execution_config.pgd_restarts,
            execution_config.pgd_steps,
            execution_config.timeout.as_secs(),
            ctx.gemm_engine,
            ctx.attack_engine_source,
            ctx.json,
            &bab_ledger,
        )
    } else {
        if !ctx.json {
            info!("Running β-CROWN...");
        }
        // Thread the ledger's wall-clock deadline (#4321) so the initial-bound
        // pass and BaB phase budgets remain inside the one dispatch budget,
        // including time spent in the IBP and structural fast paths above.
        // Without this, a large-model initial α-CROWN/CROWN pass could run past
        // the competition --timeout and get OS-killed (exit 124) before any
        // JSON verdict is written. The deadline forces a graceful Timeout/Unknown.
        verify_standard(
            &*ctx.model_net,
            ctx.input,
            ctx.effective_threshold,
            ctx.const_output_idx,
            ctx.output_dim,
            ctx.use_relu_split,
            ctx.gpu_bab,
            &verifier,
            ctx.gemm_engine,
            bab_ledger.bab_deadline(),
        )
    };
    #[cfg(feature = "mip")]
    if let Some(attachment) = cgan_input_leaf_attachment.as_ref() {
        attachment.emit_final_once("bab-return");
    }
    let result = match verification {
        Ok(result) => result,
        Err(error) => {
            let Some(timeout) = timeout_result_from_deadline_error(
                &error,
                ctx.verification_start,
                bab_ledger.bab_deadline(),
            ) else {
                return Err(error);
            };
            timeout
        }
    };

    let deadline = internal_authority_deadline;
    let confirmation_deadline =
        potential_violation_confirmation_deadline(&bab_ledger, ctx.outer_wrapper_owns_deferred_pgd);
    // Observation-only handoff marker: under NY_PHASE_TELEMETRY=1, expose
    // whether BaB actually returned while its reserved MIP slice was live.
    // This is deliberately before potential-violation confirmation so that
    // post-BaB attack work cannot obscure the phase boundary.
    bab_ledger.emit_telemetry("post-bab");
    let result = confirm_potential_violation(
        &*ctx.model_net,
        ctx.input,
        ctx.vnnlib_spec,
        result,
        &execution_config,
        confirmation_deadline,
        ctx.json,
    )?;
    let result = gate_result_at_deadline(result, ctx.verification_start, deadline);

    // Auto-escalate to the MIP complete verifier when BaB is inconclusive
    // (#4246). Under `--features mip` this fires for both an explicit
    // `--complete-verifier mip` (unconditionally, preserving the prior
    // behavior) and the default `auto` policy (gated on MIP-encodability so we
    // only escalate when the MIP pipeline can actually encode the net within the size
    // cap). Explicit `--complete-verifier bab` never escalates. Escalation is
    // sound: certificate-gated Big-M MIP can only turn unknown/timeout into
    // a decided verdict — the BaB result is kept unless it was inconclusive.
    #[cfg(feature = "mip")]
    {
        let bab_inconclusive = matches!(
            result.result,
            BabVerificationStatus::Timeout | BabVerificationStatus::Unknown { .. }
        );
        let policy_allows = matches!(
            complete_verifier,
            CompleteVerifierArg::Mip | CompleteVerifierArg::Auto
        );
        if policy_allows && bab_inconclusive {
            let initial_mip_timeout = bab_ledger.mip_timeout().unwrap_or(0);
            info!(
                "MIP escalation gate: policy_allows={policy_allows} bab_status={:?} mip_timeout={initial_mip_timeout}s remaining={}s graph_mip_enabled={}",
                result.result,
                bab_ledger.remaining_secs_clamped(),
                super::graph_mip::graph_mip_enabled()
            );
            if initial_mip_timeout >= 5 {
                // Reload a fresh sequential network for MIP encoding (the BaB
                // model may be a graph). Check encodability on the network we
                // would actually hand to the MIP encoder.
                //
                // Graph-only models (multi-output Split, DAG topology) have no
                // sequential form, so the reload/conversion can fail — that is
                // a MIP-INELIGIBILITY signal, not a run-level error. Skipping
                // escalation and reporting the (sound, inconclusive) BaB
                // verdict is strictly better than aborting the whole run with
                // `?` AFTER BaB already spent its budget (nn4sys pensieve/mscn
                // hit this: "Split has 2 outputs and requires graph network
                // construction" discarded the BaB verdict and every verdict
                // downstream of it).
                let seq_net_reload = load_onnx_with_config(ctx.model_path, ctx.onnx_load_config)
                    .map_err(anyhow::Error::from)
                    .and_then(|onnx_reload| {
                        onnx_reload
                            .to_propagate_network()
                            .map_err(anyhow::Error::from)
                    });
                // Graph-MIP escalation attempt (increment 5, default-on;
                // `NY_GRAPH_MIP=0` disables it): the DAG-aware encoder +
                // certified-ay path for graph models (cifar100/tinyimagenet
                // resnets). Eligibility
                // (layer set + unstable-ReLU binary budget) and the policy
                // gate live inside; returns `true` when the MIP path took over
                // reporting; anything else keeps the (sound, inconclusive) BaB
                // verdict — escalation can only improve, never cost, a
                // verdict. Called from EVERY sequential-arm decline point, not
                // just the no-sequential-form error: the cifar100 resnet
                // reloads to a 40-layer sequential form that is NOT
                // sequential-encodable (residual Add / BatchNorm), which
                // previously fell through silently and never reached the graph
                // encoder (measured on prop1498: reload Ok at 00:12:17 →
                // Result: timeout with no escalation attempt).
                let try_graph_escalation = || -> bool {
                    if !super::graph_mip::graph_mip_enabled() {
                        return false;
                    }
                    let Some(mip_deadline) = bab_ledger.mip_deadline() else {
                        // Unbounded AUTO verification does not manufacture a
                        // finite whole-net MIP grant. Explicit MIP-only mode is
                        // rejected at its own ingress.
                        return false;
                    };
                    let live_mip_timeout = bab_ledger.mip_timeout().unwrap_or(0);
                    if live_mip_timeout < 5 {
                        info!(
                            "Graph-MIP escalation skipped after preprocessing/reload: \
                             only {live_mip_timeout}s remain"
                        );
                        return false;
                    }
                    if let (BetaCrownModel::Graph(graph), Some(vnnlib)) =
                        (&*ctx.model_net, ctx.vnnlib_spec)
                    {
                        match super::graph_mip_escalate::try_graph_mip_escalation_with_treatment(
                            graph,
                            ctx.input,
                            vnnlib,
                            &verifier,
                            complete_verifier,
                            &result.result,
                            mip_deadline,
                            mip_solver,
                            ctx.property.as_deref(),
                            Some(ctx.model_path),
                            ctx.epsilon,
                            ctx.effective_threshold,
                            ctx.verification_start,
                            Some(ctx.effective_treatment),
                            ctx.json,
                        ) {
                            Ok(took_over) => return took_over,
                            Err(err) => {
                                warn!(
                                    "Graph-MIP escalation failed (degrading to BaB {:?}): {:#}",
                                    result.result, err
                                );
                            }
                        }
                    }
                    false
                };
                let seq_net = match seq_net_reload {
                    Ok(seq_net) => Some(Box::new(seq_net)),
                    Err(e) => {
                        if try_graph_escalation() {
                            return Ok(());
                        }
                        info!(
                            "MIP escalation ineligible (model has no sequential form; keeping BaB {:?}): {:#}",
                            result.result, e
                        );
                        // The strict graph path above is authoritative. In
                        // particular, an eligibility decline (including its
                        // 5M-NNZ memory gate) is terminal for whole-net MIP:
                        // never fall through to the legacy per-clause encoder,
                        // which has no equivalent pre-encode envelope.
                        None
                    }
                };
                if let Some(seq_net) = seq_net {
                    // Explicit `mip` keeps the prior unconditional fallback; `auto`
                    // only escalates for MIP-encodable nets within the size cap.
                    let encodable = complete_verifier == CompleteVerifierArg::Mip
                        || is_mip_encodable_sequential(&seq_net, MIP_AUTO_ESCALATION_MAX_PARAMS);
                    info!(
                        "MIP escalation: sequential reload Ok ({} layers), sequential-encodable={encodable}",
                        seq_net.layers().len()
                    );
                    if !encodable && try_graph_escalation() {
                        // Sequential form exists but is not sequential-encodable
                        // (e.g. resnet Add/BatchNorm) — the graph encoder decided it.
                        return Ok(());
                    }
                    let should_escalate =
                        should_auto_escalate_to_mip(complete_verifier, &result.result, encodable);
                    let live_mip_timeout = bab_ledger.mip_timeout().unwrap_or(0);
                    if should_escalate && live_mip_timeout < 5 {
                        info!(
                            "Sequential MIP escalation skipped after model reload: \
                             only {live_mip_timeout}s remain"
                        );
                    }
                    if should_escalate && live_mip_timeout >= 5 {
                        let Some(mip_deadline) = bab_ledger.mip_deadline() else {
                            // See the graph closure above: AUTO mode has no
                            // defensible implicit grant for an unbounded run.
                            return Ok(());
                        };
                        let escalation_kind = if complete_verifier == CompleteVerifierArg::Auto {
                            "auto-escalating"
                        } else {
                            "falling back"
                        };
                        info!(
                            "BaB inconclusive ({:?}), {} to MIP with {}s budget ({}s remaining from BaB)",
                            result.result,
                            escalation_kind,
                            live_mip_timeout,
                            bab_ledger.remaining_secs_clamped()
                        );
                        let mut seq_net = BetaCrownModel::Sequential(seq_net);
                        apply_heuristic_sound_modes(
                            &mut seq_net,
                            ctx.allow_heuristic_logsoftmax,
                            ctx.allow_heuristic_softmax,
                            ctx.json,
                        );
                        let mip_result = mip_highs::verify_with_mip(
                            &seq_net,
                            ctx.input,
                            ctx.vnnlib_spec,
                            ctx.property.as_deref(),
                            Some(ctx.model_path),
                            ctx.epsilon,
                            ctx.effective_threshold,
                            mip_deadline,
                            None,
                            mip_solver,
                            ctx.verification_start,
                            ctx.effective_treatment,
                            ctx.json,
                        );
                        match mip_result {
                            Ok(()) => return Ok(()),
                            Err(e) => {
                                // The MIP complete verifier could not produce a
                                // verdict (e.g. the Conv2d→Linear unfolding is too
                                // large to encode, a shape was missing, or an
                                // unsupported layer survived). This is NOT a wrong
                                // verdict: escalation can only ever *improve* on the
                                // BaB result, so a failed encoding degrades soundly
                                // back to the inconclusive BaB verdict
                                // (`unknown`/`timeout`) rather than surfacing a
                                // competition `error`. `verify_with_mip` reports
                                // its own decided verdicts internally before
                                // returning `Ok`, so reaching this arm means the MIP
                                // path decided nothing.
                                warn!(
                                "MIP complete verifier did not decide (degrading to BaB {:?}): {:#}",
                                result.result, e
                            );
                                // The sequential MIP couldn't even encode/decide —
                                // give the graph encoder its shot before degrading.
                                if try_graph_escalation() {
                                    return Ok(());
                                }
                                // Fall through to report the inconclusive BaB result.
                            }
                        }
                    }
                    // Not escalating: fall through and report the BaB result. `seq_net`
                    // (the unused reload) is dropped here.
                    let _ = seq_net;
                }
            }
        }
    }

    // Verdict FIRST, certificate second (see the IBP fast-path site): a future
    // emission-path pathology must never delay or mask a decided verdict.
    // Re-apply both elapsed normalization and the deadline gate here because
    // model reload and declined/failed escalation attempts above may have
    // consumed additional wall time after the first normalization.
    let result =
        gate_result_at_deadline(result, ctx.verification_start, internal_authority_deadline);
    output_result(
        &result,
        ctx.property,
        Some(ctx.model_path),
        ctx.epsilon,
        ctx.effective_threshold,
        ctx.verify_upper,
        ctx.json,
        ctx.sigmoid_peeled,
        ctx.effective_treatment,
    )?;
    super::cert_adapter::maybe_emit_certificate(ctx, &result, internal_authority_deadline);

    Ok(())
}

#[cfg(test)]
mod deadline_error_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn deferred_pgd_owner_selects_one_reservation_and_one_executor() {
        let config = BetaCrownConfig {
            enable_pgd_attack: true,
            phase_budget: ny_propagate::PhaseBudgetConfig {
                post_bab_pgd_fraction: 0.25,
                ..ny_propagate::PhaseBudgetConfig::default()
            },
            ..BetaCrownConfig::default()
        };

        let internal_policy = ledger_policy_for_deferred_pgd_owner(&config, true, false);
        assert_eq!(
            internal_policy.post_bab_pgd_fraction, 0.0,
            "the CLI must not pre-subtract the engine-owned deferred slice"
        );
        let internal_config = verifier_config_for_deferred_pgd_owner(&config, false);
        assert!(internal_config.enable_pgd_attack);
        assert_eq!(internal_config.phase_budget.post_bab_pgd_fraction, 0.25);

        let outer_policy = ledger_policy_for_deferred_pgd_owner(&config, false, true);
        assert_eq!(
            outer_policy.post_bab_pgd_fraction, 0.0,
            "the immutable context deadline already removed the outer wrapper's slice"
        );
        let outer_config = verifier_config_for_deferred_pgd_owner(&config, true);
        assert!(!outer_config.enable_pgd_attack);
        assert_eq!(
            outer_config.phase_budget.post_bab_pgd_fraction, 0.0,
            "the inner engine must neither execute nor reserve the outer-owned slice"
        );

        let outer_ledger = PhaseBudgetLedger::new(75, outer_policy).with_mip_reservation(true);
        assert!(
            outer_ledger.bab_deadline() < outer_ledger.overall_deadline(),
            "the regression must exercise a real MIP handoff inside the trimmed 75s authority"
        );
        assert_eq!(
            potential_violation_confirmation_deadline(&outer_ledger, true),
            outer_ledger.bab_deadline(),
            "confirmation must consume neither the MIP handoff nor the outer wrapper's tail"
        );
        assert_eq!(
            potential_violation_confirmation_deadline(&outer_ledger, false),
            outer_ledger.overall_deadline(),
            "without an outer owner, confirmation retains the historical overall horizon"
        );
    }

    #[test]
    fn typed_deadline_error_becomes_bab_timeout_with_context_chain() {
        let start = Instant::now()
            .checked_sub(Duration::from_millis(10))
            .expect("ten milliseconds before now is representable");
        let deadline = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond before now is representable");
        let error = anyhow::Error::new(ny_core::NyError::DeadlineExceeded(
            "ConvTranspose backward deadline".to_string(),
        ))
        .context("batched graph propagation failed");

        let result = timeout_result_from_deadline_error(&error, start, Some(deadline))
            .expect("typed deadline with an expired authority must become Timeout");

        assert!(matches!(result.result, BabVerificationStatus::Timeout));
        assert_eq!(result.domains_explored, 0);
        assert!(result.output_bounds.is_none());
        assert!(result.time_elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn ordinary_future_local_and_unbounded_errors_do_not_become_timeout() {
        let start = Instant::now();
        let future_authority = Some(Instant::now() + Duration::from_secs(1));
        let ordinary = anyhow::Error::new(ny_core::NyError::InvalidSpec(
            "malformed objective".to_string(),
        ));
        assert!(timeout_result_from_deadline_error(&ordinary, start, future_authority).is_none());

        let local_deadline = anyhow::Error::new(ny_core::NyError::DeadlineExceeded(
            "foreign local lease".to_string(),
        ));
        assert!(
            timeout_result_from_deadline_error(&local_deadline, start, future_authority).is_none(),
            "a live authority must not promote a node-local deadline to Timeout"
        );
        assert!(
            timeout_result_from_deadline_error(&local_deadline, start, None).is_none(),
            "an unbounded caller must not claim its own budget expired"
        );
    }
}

#[cfg(all(test, feature = "mip"))]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array1, Array2};
    use ny_onnx::vnnlib::parse_vnnlib;
    use ny_propagate::layers::{AddConstantLayer, LinearLayer, ReLULayer, SigmoidLayer};
    use ny_propagate::{GraphNetwork, Layer, Network, PhaseBudgetConfig};
    use std::cell::Cell;
    use std::ffi::OsStr;
    use std::time::Duration;

    fn decided_result() -> BetaCrownResult {
        BetaCrownResult {
            result: BabVerificationStatus::Verified,
            domains_explored: 7,
            domains_verified: 7,
            cuts_generated: 2,
            max_depth_reached: 3,
            time_elapsed: Duration::ZERO,
            output_bounds: None,
        }
    }

    #[test]
    fn verdict_finished_after_deadline_is_suppressed() {
        let start = Instant::now()
            .checked_sub(Duration::from_millis(10))
            .expect("ten milliseconds before now is representable");
        let deadline = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond before now is representable");
        let gated = gate_result_at_deadline(decided_result(), start, Some(deadline));
        assert!(matches!(gated.result, BabVerificationStatus::Timeout));
        assert_eq!(gated.domains_explored, 7);
        assert!(gated.output_bounds.is_none());
        assert!(gated.time_elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn verdict_before_deadline_is_retained_and_normalized() {
        let start = Instant::now()
            .checked_sub(Duration::from_millis(10))
            .expect("ten milliseconds before now is representable");
        let deadline = Instant::now() + Duration::from_secs(1);
        let gated = gate_result_at_deadline(decided_result(), start, Some(deadline));
        assert!(matches!(gated.result, BabVerificationStatus::Verified));
        assert!(gated.time_elapsed >= Duration::from_millis(10));
    }

    /// Build a small sequential Linear -> ReLU -> Linear network (MIP-encodable).
    fn small_relu_net() -> Network {
        let mut net = Network::new();
        net.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]), None).expect("linear1"),
        ));
        net.add_layer(Layer::ReLU(ReLULayer));
        net.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32, -1.0]]), None).expect("linear2"),
        ));
        net
    }

    fn safenlp_direct_first_fixture(
        hidden_dim: usize,
    ) -> (BetaCrownModel, BoundedTensor, VnnLibSpec) {
        let mut net = Network::new();
        net.add_layer(Layer::Linear(
            LinearLayer::new(Array2::zeros((hidden_dim, 2)), None).expect("input linear"),
        ));
        net.add_layer(Layer::ReLU(ReLULayer));
        net.add_layer(Layer::Linear(
            LinearLayer::new(Array2::zeros((2, hidden_dim)), None).expect("output linear"),
        ));
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -1.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .expect("finite two-dimensional box");
        let spec = parse_vnnlib(
            r#"
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (>= X_1 -1.0))
(assert (<= X_1 1.0))
(assert (<= Y_0 Y_1))
"#,
        )
        .expect("one relational SafeNLP-shaped row");
        (BetaCrownModel::Sequential(Box::new(net)), input, spec)
    }

    /// Match the sequential representation produced by the ONNX loader for
    /// SafeNLP MatMul+Add nodes: affine biases remain separate constant ops
    /// until the authoritative MIP preprocessing pipeline folds them.
    fn safenlp_direct_first_foldable_bias_fixture(
        hidden_dim: usize,
    ) -> (BetaCrownModel, BoundedTensor, VnnLibSpec) {
        let (_, input, spec) = safenlp_direct_first_fixture(hidden_dim);
        let mut net = Network::new();
        net.add_layer(Layer::Linear(
            LinearLayer::new(Array2::zeros((hidden_dim, 2)), None).expect("input MatMul"),
        ));
        net.add_layer(Layer::AddConstant(AddConstantLayer::new(
            Array1::from_elem(hidden_dim, 0.25_f32).into_dyn(),
        )));
        net.add_layer(Layer::ReLU(ReLULayer));
        net.add_layer(Layer::Linear(
            LinearLayer::new(Array2::zeros((2, hidden_dim)), None).expect("output MatMul"),
        ));
        net.add_layer(Layer::AddConstant(AddConstantLayer::new(
            arr1(&[0.125_f32, -0.25]).into_dyn(),
        )));
        (BetaCrownModel::Sequential(Box::new(net)), input, spec)
    }

    fn direct_first_plan_for_fixture(
        model: &BetaCrownModel,
        input: &BoundedTensor,
        spec: &VnnLibSpec,
        deadline: Option<Instant>,
        now: Instant,
    ) -> Option<SafeNlpDirectMipFirstPlan> {
        let (deadline, spec) = safenlp_direct_mip_first_preflight(
            true,
            true,
            CompleteVerifierArg::Auto,
            MipSolverArg::AY,
            Some(spec),
            deadline,
            now,
        )
        .ok()?;
        safenlp_direct_mip_first_plan_for_reloaded_model(model, input, spec, deadline, now).ok()
    }

    #[test]
    fn safenlp_direct_first_selects_cold_shared_prefix_on_one_absolute_deadline() {
        let (model, input, spec) = safenlp_direct_first_fixture(128);
        let now = Instant::now();
        let deadline = now + Duration::from_secs(15);
        let plan = direct_first_plan_for_fixture(&model, &input, &spec, Some(deadline), now)
            .expect("exact SafeNLP shape must admit the treatment");

        assert_eq!(plan.deadline, deadline);
        assert_eq!(plan.hidden_dim, 128);
        assert_eq!(plan.seed_policy, MipOnlySeedPolicy::ColdNoWarmStart);
        assert_eq!(plan.solver_route, MipOnlySolverRoute::SharedBinaryPrefix);
        assert!(
            !plan.seed_policy.run_internal_pgd(true),
            "the typed treatment must suppress even a configured direct-MIP PGD precheck"
        );
        assert!(
            !plan.seed_policy.permits_warm_start(),
            "the treatment must pass no warm seed, preserving shared-prefix admission"
        );

        assert!(
            MipOnlySeedPolicy::HistoricalConfigured.run_internal_pgd(true),
            "generic run_mip_only callers must retain their configured PGD behavior"
        );
        assert!(
            MipOnlySeedPolicy::HistoricalConfigured.permits_warm_start(),
            "generic run_mip_only callers must retain PGD-to-MIP warm starts"
        );
    }

    #[test]
    fn safenlp_matmul_add_reload_canonicalizes_before_admission() {
        let (model, input, spec) = safenlp_direct_first_foldable_bias_fixture(128);
        let BetaCrownModel::Sequential(network) = &model else {
            panic!("fixture must be sequential")
        };
        assert!(matches!(
            network.layers(),
            [
                Layer::Linear(_),
                Layer::AddConstant(_),
                Layer::ReLU(_),
                Layer::Linear(_),
                Layer::AddConstant(_)
            ]
        ));

        let now = Instant::now();
        let deadline = now + Duration::from_secs(15);
        let plan =
            safenlp_direct_mip_first_plan_for_reloaded_model(&model, &input, &spec, deadline, now)
                .expect("the exact MIP fold/validate/extract pipeline must admit MatMul+Add");
        assert_eq!(plan.hidden_dim, 128);
        assert_eq!(plan.deadline, deadline);
    }

    #[test]
    fn safenlp_oversized_raw_bias_declines_before_route_admission() {
        let (_, input, spec) = safenlp_direct_first_fixture(2);
        let mut net = Network::new();
        net.add_layer(Layer::Linear(
            LinearLayer::new(Array2::zeros((2, 2)), None).expect("input MatMul"),
        ));
        net.add_layer(Layer::AddConstant(AddConstantLayer::new(
            arr1(&[0.0_f32, 0.0, 0.0]).into_dyn(),
        )));
        net.add_layer(Layer::ReLU(ReLULayer));
        net.add_layer(Layer::Linear(
            LinearLayer::new(Array2::zeros((2, 2)), None).expect("output MatMul"),
        ));
        let model = BetaCrownModel::Sequential(Box::new(net));
        let now = Instant::now();
        let deadline = now + Duration::from_secs(15);

        assert!(matches!(
            safenlp_direct_mip_first_plan_for_reloaded_model(&model, &input, &spec, deadline, now),
            Err(SafeNlpDirectMipFirstDecline::UnsupportedReloadedModelShape)
        ));
    }

    #[test]
    fn safenlp_graph_context_admits_only_through_one_valid_sequential_reload() {
        let graph_context = BetaCrownModel::Graph(Box::new(GraphNetwork::new()));
        let source = SafeNlpDirectMipModelSource::from_model(&graph_context);
        assert_eq!(source, SafeNlpDirectMipModelSource::Graph);

        let (reloaded_model, input, spec) = safenlp_direct_first_foldable_bias_fixture(128);
        let now = Instant::now();
        let deadline = now + Duration::from_secs(15);
        assert!(matches!(
            safenlp_direct_mip_first_plan_for_reloaded_model(
                &graph_context,
                &input,
                &spec,
                deadline,
                now
            ),
            Err(SafeNlpDirectMipFirstDecline::UnsupportedReloadedModelShape)
        ));

        let reloads = Cell::new(0);
        let executions = Cell::new(0);
        let attempt = safenlp_direct_mip_first_attempt_with_reload(
            true,
            true,
            CompleteVerifierArg::Auto,
            MipSolverArg::AY,
            source,
            &input,
            Some(&spec),
            Some(deadline),
            now,
            || {
                reloads.set(reloads.get() + 1);
                Ok(reloaded_model)
            },
            || now + Duration::from_millis(1),
            |model, plan, observed_source| {
                executions.set(executions.get() + 1);
                assert!(matches!(model, BetaCrownModel::Sequential(_)));
                assert_eq!(observed_source, SafeNlpDirectMipModelSource::Graph);
                plan
            },
        );

        let SafeNlpDirectMipFirstAttempt::Executed(plan) = attempt else {
            panic!("valid sequential reload must admit the graph-shaped BaB context")
        };
        assert_eq!(plan.deadline, deadline);
        assert_eq!(plan.hidden_dim, 128);
        assert_eq!(
            reloads.get(),
            1,
            "the direct route must reload exactly once"
        );
        assert_eq!(
            executions.get(),
            1,
            "one admitted reload must start exactly one route/session owner"
        );
    }

    #[test]
    fn safenlp_reload_failure_wrong_shape_and_deadlines_decline_before_execution() {
        let (_, input, spec) = safenlp_direct_first_fixture(2);
        let now = Instant::now();
        let deadline = now + Duration::from_secs(15);
        let reloads = Cell::new(0);
        let executions = Cell::new(0);
        let load_failure = safenlp_direct_mip_first_attempt_with_reload(
            true,
            true,
            CompleteVerifierArg::Auto,
            MipSolverArg::AY,
            SafeNlpDirectMipModelSource::Graph,
            &input,
            Some(&spec),
            Some(deadline),
            now,
            || {
                reloads.set(reloads.get() + 1);
                anyhow::bail!("synthetic reload failure")
            },
            || now + Duration::from_millis(1),
            |_, _, _| {
                executions.set(executions.get() + 1);
            },
        );
        assert!(matches!(
            load_failure,
            SafeNlpDirectMipFirstAttempt::Declined(
                SafeNlpDirectMipFirstDecline::ReloadFailed(ref detail)
            ) if detail.contains("synthetic reload failure")
        ));
        assert_eq!(reloads.get(), 1);
        assert_eq!(executions.get(), 0);

        let (wide_model, _, _) = safenlp_direct_first_fixture(129);
        let wrong_shape = safenlp_direct_mip_first_attempt_with_reload(
            true,
            true,
            CompleteVerifierArg::Auto,
            MipSolverArg::AY,
            SafeNlpDirectMipModelSource::Graph,
            &input,
            Some(&spec),
            Some(deadline),
            now,
            || Ok(wide_model),
            || now + Duration::from_millis(1),
            |_, _, _| {
                executions.set(executions.get() + 1);
            },
        );
        assert!(matches!(
            wrong_shape,
            SafeNlpDirectMipFirstAttempt::Declined(
                SafeNlpDirectMipFirstDecline::UnsupportedReloadedModelShape
            )
        ));
        assert_eq!(executions.get(), 0);

        let preflight_reloads = Cell::new(0);
        let expired_before_reload = safenlp_direct_mip_first_attempt_with_reload(
            true,
            true,
            CompleteVerifierArg::Auto,
            MipSolverArg::AY,
            SafeNlpDirectMipModelSource::Graph,
            &input,
            Some(&spec),
            Some(now),
            now,
            || {
                preflight_reloads.set(preflight_reloads.get() + 1);
                let (model, _, _) = safenlp_direct_first_fixture(2);
                Ok(model)
            },
            || now,
            |_, _, _| {
                executions.set(executions.get() + 1);
            },
        );
        assert!(matches!(
            expired_before_reload,
            SafeNlpDirectMipFirstAttempt::Declined(SafeNlpDirectMipFirstDecline::ExpiredDeadline)
        ));
        assert_eq!(
            preflight_reloads.get(),
            0,
            "expired preflight must not touch the loader"
        );

        let (model, _, _) = safenlp_direct_first_fixture(2);
        let expired_after_reload = safenlp_direct_mip_first_attempt_with_reload(
            true,
            true,
            CompleteVerifierArg::Auto,
            MipSolverArg::AY,
            SafeNlpDirectMipModelSource::Graph,
            &input,
            Some(&spec),
            Some(deadline),
            now,
            || Ok(model),
            || deadline,
            |_, _, _| {
                executions.set(executions.get() + 1);
            },
        );
        assert!(matches!(
            expired_after_reload,
            SafeNlpDirectMipFirstAttempt::Declined(SafeNlpDirectMipFirstDecline::ExpiredDeadline)
        ));
        assert_eq!(executions.get(), 0);

        let (model, _, _) = safenlp_direct_first_foldable_bias_fixture(2);
        let deadline_samples = Cell::new(0);
        let expired_after_canonicalization = safenlp_direct_mip_first_attempt_with_reload(
            true,
            true,
            CompleteVerifierArg::Auto,
            MipSolverArg::AY,
            SafeNlpDirectMipModelSource::Graph,
            &input,
            Some(&spec),
            Some(deadline),
            now,
            || Ok(model),
            || {
                let sample = deadline_samples.get();
                deadline_samples.set(sample + 1);
                if sample == 0 {
                    now + Duration::from_millis(1)
                } else {
                    deadline
                }
            },
            |_, _, _| {
                executions.set(executions.get() + 1);
            },
        );
        assert!(matches!(
            expired_after_canonicalization,
            SafeNlpDirectMipFirstAttempt::Declined(SafeNlpDirectMipFirstDecline::ExpiredDeadline)
        ));
        assert_eq!(
            deadline_samples.get(),
            2,
            "canonicalization must be charged before route execution"
        );
        assert_eq!(executions.get(), 0);
    }

    #[test]
    fn safenlp_default_and_preflight_declines_never_reload_or_execute() {
        use crate::commands::beta_crown::output::{
            begin_capture, end_capture, take_captured_terminal_ingress, CapturedTerminalIngress,
        };

        let (_, input, spec) = safenlp_direct_first_fixture(2);
        let now = Instant::now();
        let deadline = now + Duration::from_secs(15);

        for (requested, shared, expected) in [
            (false, true, SafeNlpDirectMipFirstDecline::NotRequested),
            (
                true,
                false,
                SafeNlpDirectMipFirstDecline::SharedPrefixDisabled,
            ),
        ] {
            begin_capture();
            let reloads = Cell::new(0);
            let executions = Cell::new(0);
            let attempt = safenlp_direct_mip_first_attempt_with_reload(
                requested,
                shared,
                CompleteVerifierArg::Auto,
                MipSolverArg::AY,
                SafeNlpDirectMipModelSource::Graph,
                &input,
                Some(&spec),
                Some(deadline),
                now,
                || {
                    reloads.set(reloads.get() + 1);
                    let (model, _, _) = safenlp_direct_first_fixture(2);
                    Ok(model)
                },
                || now,
                |_, _, _| {
                    executions.set(executions.get() + 1);
                },
            );
            assert!(matches!(
                attempt,
                SafeNlpDirectMipFirstAttempt::Declined(ref reason) if *reason == expected
            ));
            assert_eq!(reloads.get(), 0);
            assert_eq!(executions.get(), 0);
            assert_eq!(
                take_captured_terminal_ingress(),
                CapturedTerminalIngress::None,
                "a pre-route decline must preserve historical post-BaB policy"
            );
            end_capture();
        }
    }

    #[test]
    fn safenlp_direct_first_declines_missing_scope_prefix_policy_or_live_deadline() {
        let (_, _, spec) = safenlp_direct_first_fixture(2);
        let now = Instant::now();
        let deadline = Some(now + Duration::from_secs(15));
        let decide = |requested, shared, policy, deadline| {
            safenlp_direct_mip_first_preflight(
                requested,
                shared,
                policy,
                MipSolverArg::AY,
                Some(&spec),
                deadline,
                now,
            )
        };

        assert!(matches!(
            decide(false, true, CompleteVerifierArg::Auto, deadline),
            Err(SafeNlpDirectMipFirstDecline::NotRequested)
        ));
        assert!(matches!(
            decide(true, false, CompleteVerifierArg::Auto, deadline),
            Err(SafeNlpDirectMipFirstDecline::SharedPrefixDisabled)
        ));
        assert!(matches!(
            decide(true, true, CompleteVerifierArg::Bab, deadline),
            Err(SafeNlpDirectMipFirstDecline::NonAutoPolicy)
        ));
        assert!(matches!(
            decide(true, true, CompleteVerifierArg::Mip, deadline),
            Err(SafeNlpDirectMipFirstDecline::NonAutoPolicy)
        ));
        assert!(matches!(
            decide(true, true, CompleteVerifierArg::Auto, None),
            Err(SafeNlpDirectMipFirstDecline::MissingDeadline)
        ));
        assert!(matches!(
            decide(true, true, CompleteVerifierArg::Auto, Some(now)),
            Err(SafeNlpDirectMipFirstDecline::ExpiredDeadline)
        ));
        assert!(decide(
            true,
            true,
            CompleteVerifierArg::Auto,
            now.checked_sub(Duration::from_millis(1)),
        )
        .is_err());
    }

    #[test]
    fn safenlp_direct_first_declines_broad_network_and_property_shapes() {
        let now = Instant::now();
        let deadline = Some(now + Duration::from_secs(15));

        let (wide_model, input, spec) = safenlp_direct_first_fixture(129);
        assert!(direct_first_plan_for_fixture(&wide_model, &input, &spec, deadline, now).is_none());

        let (mut extra_model, input, spec) = safenlp_direct_first_fixture(2);
        let BetaCrownModel::Sequential(extra_net) = &mut extra_model else {
            unreachable!()
        };
        extra_net.add_layer(Layer::ReLU(ReLULayer));
        assert!(
            direct_first_plan_for_fixture(&extra_model, &input, &spec, deadline, now).is_none()
        );

        let (model, input, _) = safenlp_direct_first_fixture(2);
        let constant_spec = parse_vnnlib(
            r#"
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (>= X_1 -1.0))
(assert (<= X_1 1.0))
(assert (<= Y_0 0.0))
"#,
        )
        .expect("constant-row property");
        assert!(
            direct_first_plan_for_fixture(&model, &input, &constant_spec, deadline, now).is_none()
        );

        let mut multi_spec = spec;
        multi_spec
            .output_constraints
            .push(OutputConstraint::LessEq(1, 0));
        multi_spec.output_constraint_clauses = vec![multi_spec.output_constraints.clone()];
        assert!(
            direct_first_plan_for_fixture(&model, &input, &multi_spec, deadline, now).is_none()
        );
    }

    #[test]
    fn safenlp_direct_first_phase_marker_is_exact_and_default_dark() {
        for raw in [None, Some(""), Some("0"), Some("true"), Some(" 1")] {
            assert!(!phase_telemetry_enabled_from_value(raw.map(OsStr::new)));
        }
        assert!(phase_telemetry_enabled_from_value(Some(OsStr::new("1"))));
        assert_eq!(safenlp_direct_first_marker_if(false, "route-start"), None);
        assert_eq!(
            safenlp_direct_first_marker_if(true, "route-start").as_deref(),
            Some(
                "NY_MIP_SAFENLP_DIRECT_FIRST_V1 event=route-start \
                 route=direct-mip-first shared_prefix=required internal_pgd=false \
                warm_start=false deadline=caller-absolute"
            )
        );
        assert_eq!(
            safenlp_direct_first_decline_marker_if(
                true,
                &SafeNlpDirectMipFirstDecline::NotRequested
            ),
            None,
            "default/unset/malformed/zero intent must emit no direct-route telemetry"
        );
        assert_eq!(
            safenlp_direct_first_decline_marker_if(
                false,
                &SafeNlpDirectMipFirstDecline::SharedPrefixDisabled
            ),
            None
        );
        assert_eq!(
            safenlp_direct_first_decline_marker_if(
                true,
                &SafeNlpDirectMipFirstDecline::SharedPrefixDisabled
            )
            .as_deref(),
            Some(
                "NY_MIP_SAFENLP_DIRECT_FIRST_V1 event=route-decline \
                 stage=preflight reason=shared-prefix-disabled historical_fallback=bab"
            )
        );
    }

    #[test]
    fn safenlp_direct_first_is_structurally_after_ibp_and_before_bab() {
        let source = include_str!("dispatch.rs");
        let (_, body) = source
            .split_once("pub(super) fn run_bab_with_fallback")
            .expect("production dispatch function");
        let ibp = body
            .find("if ibp_check_vnnlib_safe")
            .expect("authoritative IBP fast path");
        let affine = body
            .find("try_affine_root_farkas")
            .expect("exact affine root-Farkas closure");
        let direct = body
            .find("safenlp_direct_mip_first_attempt_with_reload")
            .expect("direct-first admission");
        let reload = body
            .find("reload_safenlp_direct_mip_model")
            .expect("fresh sequential reload");
        let bab = body
            .find("let mut verifier = match")
            .expect("BaB verifier construction");

        assert!(
            ibp < affine && affine < direct && direct < reload && reload < bab,
            "exact affine/direct-first routes must preserve IBP and run before actual BaB"
        );
        assert_eq!(
            body[..bab]
                .matches("reload_safenlp_direct_mip_model")
                .count(),
            1,
            "the pre-BaB direct path must own exactly one reload call"
        );
    }

    #[cfg(feature = "mip")]
    #[test]
    fn affine_root_farkas_preserves_explicit_bab_opt_out() {
        assert!(affine_root_farkas_policy_allows(
            CompleteVerifierArg::Auto,
            MipSolverArg::AY
        ));
        assert!(affine_root_farkas_policy_allows(
            CompleteVerifierArg::Mip,
            MipSolverArg::AY
        ));
        assert!(!affine_root_farkas_policy_allows(
            CompleteVerifierArg::Bab,
            MipSolverArg::AY
        ));
    }

    #[test]
    fn safenlp_reload_seam_wraps_sequential_and_applies_sound_modes_once() {
        let source = include_str!("dispatch.rs");
        let (_, reload_body) = source
            .split_once("fn reload_safenlp_direct_mip_model")
            .expect("production reload helper");
        let reload_body = reload_body
            .split_once("fn phase_telemetry_enabled_from_value")
            .expect("end of reload helper")
            .0;

        assert_eq!(reload_body.matches("load_onnx_with_config").count(), 1);
        assert_eq!(reload_body.matches(".to_propagate_network()").count(), 1);
        assert_eq!(
            reload_body
                .matches("BetaCrownModel::Sequential(Box::new(seq_net))")
                .count(),
            1
        );
        assert_eq!(
            reload_body.matches("apply_heuristic_sound_modes").count(),
            1
        );
    }

    #[test]
    fn encodability_accepts_small_linear_relu_net() {
        let net = small_relu_net();
        assert!(
            is_mip_encodable_sequential(&net, MIP_AUTO_ESCALATION_MAX_PARAMS),
            "a small sequential Linear+ReLU net must be MIP-encodable"
        );
    }

    #[test]
    fn encodability_rejects_unsupported_activation() {
        // A Sigmoid layer is NOT encodable by the Big-M MIP path: escalating
        // for it would make verify_with_mip bail, so the policy must refuse.
        let mut net = Network::new();
        net.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("linear"),
        ));
        net.add_layer(Layer::Sigmoid(SigmoidLayer));
        assert!(
            !is_mip_encodable_sequential(&net, MIP_AUTO_ESCALATION_MAX_PARAMS),
            "a net with a Sigmoid layer must NOT be treated as MIP-encodable"
        );
    }

    #[test]
    fn encodability_rejects_oversized_net() {
        // Encodable layer types but over the parameter cap: stay with the BaB
        // verdict instead of launching an intractable MIP.
        let net = small_relu_net();
        assert!(
            !is_mip_encodable_sequential(&net, 1),
            "a net exceeding the parameter cap must NOT be escalated"
        );
    }

    #[test]
    fn policy_auto_escalates_on_inconclusive_encodable() {
        // Default `auto` + inconclusive BaB + encodable net => escalate.
        for status in [
            BabVerificationStatus::Timeout,
            BabVerificationStatus::Unknown {
                reason: "domain cap".to_string(),
            },
        ] {
            assert!(
                should_auto_escalate_to_mip(CompleteVerifierArg::Auto, &status, true),
                "auto must escalate on inconclusive ({status:?}) encodable nets"
            );
        }
    }

    #[test]
    fn policy_auto_does_not_escalate_when_not_encodable() {
        // Even inconclusive, a non-encodable net is never escalated under `auto`:
        // we keep the BaB unknown/timeout rather than bail in MIP preprocessing.
        assert!(!should_auto_escalate_to_mip(
            CompleteVerifierArg::Auto,
            &BabVerificationStatus::Timeout,
            false,
        ));
    }

    #[test]
    fn policy_keeps_decided_bab_verdict() {
        // A decided BaB verdict (Verified/Violated/PotentialViolation) is NEVER
        // discarded — soundness invariant: escalation only fills in unknowns.
        for status in [
            BabVerificationStatus::Verified,
            BabVerificationStatus::Violated {
                counterexample: vec![0.0],
                output: vec![0.0],
            },
            BabVerificationStatus::potential_violation(),
        ] {
            for cv in [CompleteVerifierArg::Auto, CompleteVerifierArg::Mip] {
                assert!(
                    !should_auto_escalate_to_mip(cv, &status, true),
                    "decided BaB verdict ({status:?}) must be kept under {cv:?}"
                );
            }
        }
    }

    #[test]
    fn policy_explicit_bab_never_escalates() {
        // `--complete-verifier bab` opts out of escalation entirely, even when
        // inconclusive on an encodable net (backward-compatible behavior).
        assert!(!should_auto_escalate_to_mip(
            CompleteVerifierArg::Bab,
            &BabVerificationStatus::Timeout,
            true,
        ));
    }

    #[test]
    fn policy_explicit_mip_escalates_on_inconclusive() {
        // Explicit `--complete-verifier mip` keeps its prior fallback behavior.
        assert!(should_auto_escalate_to_mip(
            CompleteVerifierArg::Mip,
            &BabVerificationStatus::Unknown {
                reason: "cap".to_string()
            },
            true,
        ));
    }

    #[test]
    fn safenlp_shared_prefix_budget_gate_accepts_only_literal_one() {
        assert!(safenlp_shared_prefix_budget_repair_enabled_from_value(
            Some(OsStr::new("1"))
        ));
        for raw in [None, Some("0"), Some(" 1"), Some("1 "), Some("true")] {
            assert!(
                !safenlp_shared_prefix_budget_repair_enabled_from_value(raw.map(OsStr::new)),
                "non-exact gate spelling {raw:?} must preserve the historical schedule"
            );
        }
    }

    fn safenlp_official_internal_policy() -> PhaseBudgetConfig {
        PhaseBudgetConfig {
            mip_min_fraction: 0.65,
            mip_min_secs: 8,
            mip_max_secs: 30,
            post_bab_pgd_fraction: 0.10,
            ..PhaseBudgetConfig::default()
        }
    }

    #[test]
    fn production_budget_dispatch_is_identical_when_shared_prefix_gate_is_off() {
        assert!(
            std::env::var_os("NY_GRAPH_MIP").is_none(),
            "run this production-policy regression with NY_GRAPH_MIP unset"
        );
        let base = PhaseBudgetLedger::new(15, safenlp_official_internal_policy());
        let historical = base
            .clone()
            .with_mip_escalation_allowed(true)
            .with_static_mip_ineligibility(true);
        let routed = apply_bab_mip_budget_policy(base, CompleteVerifierArg::Auto, true, false);

        assert_eq!(routed.bab_deadline(), historical.bab_deadline());
        assert_eq!(
            routed.planned_mip_timeout_at_bab_deadline(),
            historical.planned_mip_timeout_at_bab_deadline()
        );
        assert_eq!(
            routed.mip_reservation_armed_for_test(),
            historical.mip_reservation_armed_for_test()
        );
    }

    #[test]
    fn production_budget_dispatch_admits_shared_prefix_mip_but_not_explicit_bab() {
        assert!(
            std::env::var_os("NY_GRAPH_MIP").is_none(),
            "run this production-policy regression with NY_GRAPH_MIP unset"
        );
        let repaired = apply_bab_mip_budget_policy(
            PhaseBudgetLedger::new(15, safenlp_official_internal_policy()),
            CompleteVerifierArg::Auto,
            true,
            true,
        );
        assert!(
            repaired.mip_reservation_armed_for_test(),
            "sequential reload reachability must survive graph-static ineligibility"
        );
        assert!(
            repaired
                .planned_mip_timeout_at_bab_deadline()
                .expect("bounded repair grant")
                >= 5,
            "the exact production dispatch threshold must be reachable"
        );

        let bab_only = apply_bab_mip_budget_policy(
            PhaseBudgetLedger::new(15, safenlp_official_internal_policy()),
            CompleteVerifierArg::Bab,
            true,
            true,
        );
        assert!(
            !bab_only.mip_reservation_armed_for_test(),
            "explicit complete-verifier=bab must remain no-MIP under the gate"
        );
    }

    #[cfg(feature = "mip")]
    #[test]
    fn strict_graph_mip_decline_has_no_legacy_encoder_fallback() {
        // The strict path owns the NNZ admission decision. Keep this source
        // guard so a future cleanup cannot accidentally restore the old
        // second call after a 5M-NNZ decline and bypass the memory envelope.
        let dispatch = include_str!("dispatch.rs");
        let strict_call = [
            "super::graph_mip_escalate::",
            "try_graph_mip_escalation_with_treatment(",
        ]
        .concat();
        let legacy_call = ["super::graph_mip::", "try_graph_mip_escalation("].concat();
        assert_eq!(
            dispatch.matches(&strict_call).count(),
            1,
            "dispatch must have one authoritative strict Graph-MIP call"
        );
        assert!(
            !dispatch.contains(&legacy_call),
            "a strict eligibility decline must fall back to BaB, not an ungated encoder"
        );
    }
}
