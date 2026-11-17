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

/// #mip-demand — is the whole-net Graph-MIP escalation STATICALLY refused by
/// its own encode-nnz cap on this model?
///
/// The SECOND static predicate in the escalation gate, and the one that
/// actually binds on the measured families. `#deadlane` above answers the
/// LAYER-SET predicate; this one answers `estimate_encode_nnz(..) > cap`, which
/// the gate reaches ~2 ms after BaB has already surrendered a fifth to a third
/// of its own search to fund it (`reports/measured-2026/stop_census_2026-08-30/`:
/// `challenging_certified` idx1031 `nnz 150936750`, `metaroom_2023` idx125
/// `nnz 50355927`, both against a 5,000,000 cap; the MIP slice alone is 28 s of
/// challenging's 114 s solver deadline, so BaB goes 74.6 s -> 102.6 s).
///
/// `true` requires a PROOF, never a guess: `static_encode_nnz_floor` is a lower
/// bound on the runtime estimate for ANY bounds map, so `floor > cap` implies
/// the runtime comparison the gate will make also exceeds the cap. `None` (no
/// static claim) and a floor at or under the cap both return `false`, i.e.
/// "reserve exactly as before".
///
/// Sequential models are left alone: the nnz estimator is defined on the graph
/// form, and the sequential lane has its own parameter cap.
#[cfg_attr(not(feature = "mip"), allow(unused_variables))]
pub(super) fn graph_mip_statically_over_nnz_cap(
    model: &BetaCrownModel,
    input_shape: &[usize],
) -> bool {
    #[cfg(feature = "mip")]
    {
        match model {
            BetaCrownModel::Graph(graph) => {
                match super::graph_mip_leaf::static_encode_nnz_floor(graph, input_shape) {
                    Some(floor) => floor > super::graph_mip_escalate::graph_mip_max_nnz(),
                    None => false,
                }
            }
            BetaCrownModel::Sequential(_) => false,
        }
    }
    #[cfg(not(feature = "mip"))]
    {
        false
    }
}

/// #reserve-demand — the purely STATIC half of the post-BaB falsification
/// lane's admission test: can this net take a gradient step at all?
///
/// `PhaseBudgetLedger::bab_deadline` subtracts `post_bab_pgd_fraction` from
/// BaB's slice BEFORE BaB runs, so the subtraction can only be justified by a
/// capability known BEFORE BaB. `postbab_lane_has_executable_stage`
/// (commands/vnncomp.rs) admits the lane on EITHER a seed or a gradient; a seed
/// is produced by the verifier itself and therefore cannot justify a
/// pre-payment, while `supports_attack_point_gradient` is a pure function of the
/// loaded graph and can.
///
/// `true` means every gradient stage (Newton equality-seek, DLR-APGD) is dead
/// on this net, so the reserve's only pre-BaB-knowable spender does not exist.
/// The lane is NOT disabled: if the verifier does export a seed, the
/// derivative-free stages still receive whatever wall remains after BaB — they
/// are simply not pre-paid out of BaB's slice.
///
/// Sequential models keep the historical reservation (fail safe): the
/// exact-gradient fragment test is defined on the graph form.
pub(super) fn postbab_attack_statically_ineligible(model: &BetaCrownModel) -> bool {
    match model {
        BetaCrownModel::Graph(graph) => !graph.supports_attack_point_gradient(),
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

/// #safenlp-cycle — ONE exact default-off gate for the SafeNLP full-cycle A/B
/// (`NY_SAFENLP_MIP_CYCLE=1`), composing four existing pieces without any new
/// mechanism of its own:
///
/// 1. it arms the MIP-handoff budget treatment (same effect as
///    `NY_MIP_SAFENLP_SHARED_PREFIX=1`), so the sequential reduced-verification
///    lane actually receives a BaB deadline and the planned handoff is MADE;
/// 2. it makes that handoff CONDITIONAL: the phase ledger keeps BaB's
///    HISTORICAL floor-protected slice (`max(total*(1-post_bab_pgd), total/2)`)
///    instead of the enforce_mip_handoff floor-drop that cut BaB to ~4.5s on
///    every row, and the MIP fires only in the remaining reserved wall — the
///    slice where today nothing runs (the handoff "planned and never made",
///    configs/vnncomp26/safenlp_2024.yaml);
/// 3. it defaults the sequential MIP to SERIAL (`parallel_split = 1`, the
///    Gurobi-winning one-model shape) inside `mip_highs`;
/// 4. it arms AY's verdict-preserving margin reframe (marked-margin ingress →
///    `MarginMode::Required` in the pinned ay-milp, whose threshold-stop banks
///    UNSAT on a strict bound crossing with original-model proof replay).
///
/// Measurement-only (the A/B in docs/SAFENLP_CYCLE_2026-08-20.md); a winning
/// cycle ships as typed preset keys per the campaign rule. The gate is a
/// DECLARED lever (ratchet discipline): exact `1` arms, everything else is a
/// recorded rejection resolving to the declaration's `false` default.
///
/// DELIVERY SEAM (#lever-reachability, 2026-08-23). The comment above says "a
/// winning cycle ships as typed preset keys per the campaign rule" — but the
/// A/B itself could only ever run by hand, because `run_instance.sh` exports
/// no `NY_*`. `NY_SAFENLP_MIP_CYCLE` still wins in both directions;
/// absent, the typed `bab.safenlp_mip_cycle` key decides, so the winning arm is
/// now shippable to `safenlp_2024` alone instead of needing a global default.
pub(in crate::commands::beta_crown) fn safenlp_mip_cycle_enabled() -> bool {
    crate::preset_gates::safenlp_mip_cycle_armed()
}

fn safenlp_shared_prefix_budget_repair_enabled() -> bool {
    #[cfg(feature = "mip")]
    {
        safenlp_shared_prefix_budget_repair_enabled_from_value(
            std::env::var_os("NY_MIP_SAFENLP_SHARED_PREFIX").as_deref(),
        )
        // #safenlp-cycle: the full-cycle gate needs a reachable handoff; it
        // arms the SAME budget treatment, with the conditional-handoff floor
        // preserved by the ledger (see `PhaseBudgetLedger::bab_deadline`).
        || safenlp_mip_cycle_enabled()
    }
    #[cfg(not(feature = "mip"))]
    {
        false
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
    postbab_attack_statically_ineligible: bool,
    graph_mip_statically_over_nnz_cap: bool,
    graph_mip_min_slice_secs: Option<u64>,
) -> PhaseBudgetLedger {
    ledger
        .with_mip_escalation_allowed(!matches!(complete_verifier, CompleteVerifierArg::Bab))
        .with_safenlp_shared_prefix_budget_repair(safenlp_shared_prefix_budget_repair)
        .with_static_mip_ineligibility(graph_mip_statically_ineligible)
        // #mip-demand: the two predicates below sit AFTER the SafeNLP
        // treatment for the same reason the layer-set one does — an armed
        // shared-prefix ledger keeps its slice for the later SEQUENTIAL reload,
        // which neither the graph nnz probe nor the graph slice floor describes.
        .with_static_mip_nnz_ineligibility(graph_mip_statically_over_nnz_cap)
        .with_graph_mip_min_slice_secs(graph_mip_min_slice_secs)
        .with_static_postbab_attack_ineligibility(postbab_attack_statically_ineligible)
}

/// #mip-demand — the whole-net Graph-MIP encode's minimum slice, but ONLY for a
/// model whose escalation would actually be that lane.
///
/// `graph_mip_slice_admitted` lives in the GRAPH escalation driver. The
/// sequential MIP lane has no such floor (safenlp's 15 s internal tier really
/// does hand a 5 s reserve to a solver that uses it), so a sequential model
/// must return `None` and keep the historical reservation.
fn graph_mip_whole_net_min_slice_secs(model: &BetaCrownModel) -> Option<u64> {
    #[cfg(feature = "mip")]
    {
        match model {
            BetaCrownModel::Graph(_) => Some(super::graph_mip_escalate::graph_mip_min_slice_secs()),
            BetaCrownModel::Sequential(_) => None,
        }
    }
    #[cfg(not(feature = "mip"))]
    {
        let _ = model;
        None
    }
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
    pub(super) input_head_continuation:
        Option<&'a crate::commands::safenlp_input_head::BoundInputHeadContinuation>,
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
    #[cfg(feature = "mip")]
    pub(super) verification_resources:
        Option<Arc<super::hull_publication::InstanceVerificationResources>>,
    /// VNN-COMP's independently resolved outer post-BaB attack route.
    /// `None` preserves the historical interactive small-budget reserve.
    pub(super) post_bab_wrapper_attack_enabled: Option<bool>,
    /// #attack-concurrent: the wrapper's admission facts for the concurrent
    /// attack lane (`None` on interactive calls: every race refuses).
    pub(super) attack_concurrent_wrapper:
        Option<super::attack_concurrent::ConcurrentAttackWrapperFacts>,
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
///
/// #admission-precision: the check is the router's
/// `ny_core::router::admission::relational_verifier` on the property's routing
/// shape (`spec_shape::shape_of`, the one extractor the router reads at start
/// too). `has_relational` is the caller's `classify_constraints` answer for
/// the same property, which is the shape's own `relational_atoms`.
pub(super) fn routes_to_relational_verifier(
    vnnlib_spec: Option<&VnnLibSpec>,
    has_relational: bool,
) -> bool {
    vnnlib_spec.is_some_and(|vnnlib| {
        let shape = super::spec_shape::shape_of(vnnlib);
        ny_core::router::admission::relational_verifier(&ny_core::router::spec::SpecShape {
            relational_atoms: has_relational,
            ..shape
        })
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

/// Bridge only a typed engine-global timeout source into wrapper capture.
/// Status and elapsed time alone cannot distinguish a phase-local timeout or
/// a natural result that merely unwound after the configured wall.
fn mark_captured_engine_timeout_source(result: &BetaCrownResult, source_deadline: Option<Instant>) {
    if matches!(result.result, BabVerificationStatus::Timeout) {
        super::output::mark_captured_timeout_at_deadline(source_deadline);
    }
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
    // #phase-cost: see `ny_core::phase_cost` (observability only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::Mip);
    let model_net: &BetaCrownModel = &*ctx.model_net;
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

    let warm_start_candidate = if ctx.run_upfront_pgd {
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
    // gates.
    mip_highs::verify_with_mip(
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
    )
}

/// The early graph verdict needs certified bounds from this complete source
/// and input. Plain graph IBP is an ordinary f32 forward and is not a proof.
fn early_graph_ibp_output(
    graph: &ny_propagate::GraphNetwork,
    input: &BoundedTensor,
    deadline: Option<Instant>,
) -> ny_core::Result<BoundedTensor> {
    let mut bounds = graph.collect_node_bounds_with_engine_and_deadline(input, None, deadline)?;
    let output = bounds.remove(graph.output_name()).ok_or_else(|| {
        ny_core::NyError::InvalidSpec("early certified graph IBP has no completed output".into())
    })?;
    if deadline.is_some_and(|end| Instant::now() >= end) {
        return Err(ny_core::NyError::DeadlineExceeded(
            "early certified graph IBP expired before output publication".into(),
        ));
    }
    Ok(output)
}

fn early_sequential_ibp_output(
    network: &ny_propagate::Network,
    input: &BoundedTensor,
    deadline: Option<Instant>,
) -> ny_core::Result<BoundedTensor> {
    let output = network
        .collect_ibp_bounds_with_deadline(input, deadline)?
        .pop()
        .ok_or_else(|| {
            ny_core::NyError::InvalidSpec(
                "early certified sequential IBP has no completed output".into(),
            )
        })?;
    if deadline.is_some_and(|end| Instant::now() >= end) {
        return Err(ny_core::NyError::DeadlineExceeded(
            "early certified sequential IBP expired before output publication".into(),
        ));
    }
    Ok(output)
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
        postbab_attack_statically_ineligible(&*ctx.model_net),
        // #mip-demand: the escalation gate's own encode-nnz cap, answered from
        // the model and the input SHAPE before BaB starts instead of ~2 ms
        // after BaB has surrendered a quarter to a third of the scored clock.
        graph_mip_statically_over_nnz_cap(&*ctx.model_net, ctx.input.shape()),
        graph_mip_whole_net_min_slice_secs(&*ctx.model_net),
    );
    #[cfg(feature = "mip")]
    let bab_ledger = bab_ledger.with_verification_resources(ctx.verification_resources.clone());
    bab_ledger.emit_telemetry("bab-dispatch");
    // #mip-demand: the lever's OWN observable, on a channel that survives into
    // the banked sidecar with no `NY_*` set — the `[budget]` line above is dark
    // behind `NY_PHASE_TELEMETRY=1` and `ny benchmarks run` forces `RUST_LOG=error`,
    // so a flight note is the only proof a future bank can carry that this
    // fired. Recorded on EVERY row, both dispositions, so a silent note is
    // distinguishable from an absent one.
    {
        let finding = bab_ledger.mip_statically_ineligible();
        crate::flight::note(
            "mip_reserve_demand",
            if finding == "none" {
                crate::flight::FlightStatus::Ran
            } else {
                crate::flight::FlightStatus::Skipped
            },
            Some(format!(
                "mip_statically_ineligible={finding} mip_reserve_fraction={:.3} \
                 demand_sized_mip_reserve={}",
                bab_ledger.mip_reserve_fraction(),
                bab_ledger.policy().demand_sized_mip_reserve,
            )),
        );
    }
    let internal_authority_deadline = bab_ledger.overall_deadline();
    if let Some(vnnlib) = ctx.vnnlib_spec {
        let ibp_result = match &*ctx.model_net {
            BetaCrownModel::Sequential(network) => {
                early_sequential_ibp_output(network, ctx.input, internal_authority_deadline)
            }
            BetaCrownModel::Graph(graph) => {
                early_graph_ibp_output(graph, ctx.input, internal_authority_deadline)
            }
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
    // Only the actual lower-bound sequential input-split lane can consume this
    // carrier. The original property/root binding is checked here; the private
    // producer also checks the complete numeric network/reduction on every
    // worker call. Graph, upper-bound, transformed and standard scalar routes
    // receive no substitute proof authority.
    if matches!(&*ctx.model_net, BetaCrownModel::Sequential(_))
        && !execution_config.verify_upper_bound
        && matches!(
            execution_config.branching_heuristic,
            ny_propagate::BranchingHeuristic::InputSplit
        )
        && !ctx.sigmoid_peeled
    {
        if let (Some(owned), Some(spec)) = (ctx.input_head_continuation, ctx.vnnlib_spec) {
            if let Some(source) = owned.admitted_source(spec, ctx.input) {
                verifier = verifier.with_input_head_continuation(source);
            }
        }
    }
    if let Some(path) = ctx.input_split_metrics_jsonl {
        verifier =
            verifier.with_input_split_metrics_sink(JsonlInputSplitMetricsSink::create(path)?);
    }
    if let Some(path) = ctx.domain_batch_metrics_jsonl {
        verifier = verifier
            .with_graph_domain_batch_metrics_sink(JsonlGraphDomainBatchMetricsSink::create(path)?);
    }
    // #vit-margin-alpha-complex (P8): attach the ONNX Runtime forward of the
    // ORIGINAL model file that adjudicates a root-terminal multi-objective
    // `Verified`. Only when the typed key is armed (no shipped preset sets it;
    // only vit_2023 may); the fence consults it only when
    // `bab.mo_containment_fence` is armed too, and it can only downgrade.
    if execution_config.mo_root_ort_adjudication {
        verifier = verifier.with_root_adjudication_oracle(Arc::new(
            super::root_ort_adjudication::OrtRootAdjudicationOracle::new(ctx.model_path),
        ));
    }
    // #relusplitter-lp-refine (wave-6 C7): attach the rigorous LP root
    // intermediate refiner only when `bab.mip_refine_intermediates` is armed
    // (no shipped preset sets it). The multi-objective graph root consults it
    // once and intersects its certified boxes shrink-only; it prints its own
    // `[mip-refine] root ...` line. This attach-time line is unconditional on
    // every armed run, so an armed key that never reached a refiner (no `mip`
    // feature) reads as `attached=0`, never as a missing line.
    if let Some(refine) = execution_config.mip_refine_intermediates {
        #[cfg(feature = "mip")]
        {
            verifier = verifier.with_root_interm_refiner(Arc::new(
                super::lp_refine_interm::LpRootIntermRefiner::new(refine),
            ));
        }
        eprintln!(
            "{}",
            ny_propagate::beta_crown::root_interm_refine::mip_refine_armed_line(
                &refine,
                cfg!(feature = "mip"),
                if cfg!(feature = "mip") {
                    "ok"
                } else {
                    "mip_feature_off"
                },
            )
        );
    }

    if let BetaCrownModel::Graph(graph) = &mut *ctx.model_net {
        graph.set_use_patches_mode(execution_config.use_patches());
        graph.set_chunk_wave_parallel_enabled(execution_config.chunk_wave_parallel_enabled);
        graph.set_frozen_levers(execution_config.frozen_levers.clone());
    }

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

    ny_propagate::beta_crown::reset_global_timeout_deadline_source();
    ny_propagate::beta_crown::reset_sequential_initial_schedule_receipt();
    ny_propagate::zono_core::reset_cgan_child_zono_receipt();
    // #attack-concurrent: the conjunctive upfront sites' race context, built
    // only when the key is armed (`None` otherwise; the sites never read it).
    // The publish hook is THIS dispatch's own Violated emission — the deadline
    // gate and the captured-route JSON renderer `output_result` uses — run on
    // the attack thread, which then ENTERS the attack door and hands the OWNED
    // JSON to the scored publisher (it replays the wrapper's sat pipeline) as
    // a job the race runs on a detached thread, so a slow gate cannot hold the
    // race's join.
    let attack_concurrent_property = ctx.property.as_deref();
    let attack_concurrent_epsilon = ctx.epsilon;
    let attack_concurrent_threshold = ctx.effective_threshold;
    let attack_concurrent_treatment = ctx.effective_treatment;
    let attack_concurrent_start = ctx.verification_start;
    let attack_concurrent_publish = |result: &BetaCrownResult,
                                     report: &super::attack_concurrent::ConcurrentSatReport|
     -> super::attack_concurrent::PreparedSat {
        use super::attack_concurrent::{ConcurrentSatPublication as Declined, PreparedSat};
        let result = gate_result_at_deadline(
            result.clone(),
            attack_concurrent_start,
            internal_authority_deadline,
        );
        if !matches!(result.result, BabVerificationStatus::Violated { .. }) {
            return PreparedSat::Declined(Declined::SuppressedAtDeadline);
        }
        match super::output::render_captured_competition_json(
            &result,
            attack_concurrent_property,
            attack_concurrent_epsilon,
            attack_concurrent_threshold,
            attack_concurrent_treatment,
        ) {
            // Enters the attack door HERE, on the attack thread inside the
            // race scope, so the main thread's synchronized door sees the
            // publication in flight however early the race stops waiting.
            Ok(json) => crate::commands::vnncomp::prepare_concurrent_attack_sat(json, *report),
            // Not adjudicated here: hand the candidate to the main thread,
            // whose own emission then renders it exactly as today.
            Err(error) => {
                tracing::warn!("#attack-concurrent: candidate JSON render failed: {error:#}");
                PreparedSat::Declined(Declined::NotArmed)
            }
        }
    };
    let attack_concurrent_ctx =
        execution_config
            .attack_concurrent
            .then(|| super::attack_concurrent::ConcurrentAttackCtx {
                wrapper: ctx.attack_concurrent_wrapper,
                publisher_armed: crate::commands::vnncomp::concurrent_attack_publisher_armed(),
                publish: &attack_concurrent_publish,
            });
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
            attack_concurrent_ctx.as_ref(),
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
        .map(super::verification_outcome::VerificationOutcome::ordinary)
    };
    let engine_timeout_source = ny_propagate::beta_crown::take_global_timeout_deadline_source();
    // #seq-single-reserve / #seq-early-attack: the levers' OWN observable, on
    // the channel that survives into the banked sidecar with no `NY_*` set —
    // the engine's `[seq-budget]` info line is dark under the RUST_LOG=error
    // that `ny benchmarks run` forces, so this flight note is the only proof
    // a future bank can carry. Recorded on EVERY row, both dispositions
    // (`Skipped` marks a row whose verification never entered the sequential
    // initial phase), so a silent note is distinguishable from an absent one.
    {
        use ny_propagate::SequentialEarlyQuitDisposition;
        match ny_propagate::take_sequential_initial_schedule_receipt() {
            Some(receipt) => crate::flight::note(
                "seq_initial_schedule",
                crate::flight::FlightStatus::Ran,
                Some(format!(
                    "effective_total_s={:.3} bab_timeout_s={:.3} warmup_cap_s={:.3} \
                     warmup_single_reserve_applied={} caller_deadline={} early_quit={}",
                    receipt.effective_total_secs,
                    receipt.bab_timeout_secs,
                    receipt.warmup_cap_secs,
                    receipt.warmup_single_reserve_applied,
                    receipt.caller_deadline,
                    match receipt.early_quit {
                        SequentialEarlyQuitDisposition::None => "none",
                        SequentialEarlyQuitDisposition::Attacked => "attacked",
                        SequentialEarlyQuitDisposition::ForfeitedByKillSwitch =>
                            "forfeited-by-kill-switch",
                    },
                )),
            ),
            None => crate::flight::note(
                "seq_initial_schedule",
                crate::flight::FlightStatus::Skipped,
                Some("lane=not-sequential-initial-phase".to_string()),
            ),
        }
    }
    // #cgan-child-zono: run totals of the per-child zonotope prepass, on the
    // channel that survives into the banked sidecar. Only an ARMED run leaves
    // a receipt, so a dark run adds no note (its sidecar is unchanged).
    if let Some(receipt) = ny_propagate::zono_core::take_cgan_child_zono_receipt() {
        crate::flight::note(
            "cgan_child_zono",
            crate::flight::FlightStatus::Ran,
            Some(receipt.render()),
        );
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
            super::verification_outcome::VerificationOutcome::ordinary(timeout)
        }
    };
    #[cfg(feature = "mip")]
    let result = {
        let (report, hull) = result.into_parts();
        if let Some(proof) = hull {
            let report = gate_result_at_deadline(
                report,
                ctx.verification_start,
                internal_authority_deadline,
            );
            return super::output::output_hull_result(
                report,
                proof,
                ctx.property.as_deref(),
                ctx.effective_threshold,
                ctx.json,
                ctx.effective_treatment,
            );
        }
        report
    };
    #[cfg(not(feature = "mip"))]
    let result = result.into_ordinary()?;
    mark_captured_engine_timeout_source(&result, engine_timeout_source);

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
                        let mip_result = mip_highs::verify_with_mip_for_bridge(
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
                            ctx.input_head_continuation.is_some(),
                        );
                        match mip_result {
                            Ok(publication) => {
                                // Only an actual typed Verified + successful
                                // MIP publication can preserve this independent
                                // full-root route after dependent source use.
                                if let (
                                    Some(owned),
                                    Some(spec),
                                    BetaCrownModel::Sequential(original),
                                    Some(publication),
                                ) = (
                                    ctx.input_head_continuation,
                                    ctx.vnnlib_spec,
                                    &seq_net,
                                    publication,
                                ) {
                                    let _ = owned.note_independent_full_root_mip_publication(
                                        original,
                                        spec,
                                        ctx.input,
                                        publication,
                                    );
                                }
                                return Ok(());
                            }
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
    // #stop-accounting: `GraphBabLifecycle::unresolved_reason()` builds the exact
    // stop reason ("Max depth 100 reached", "Queue cap ... evicted unverified
    // domains", "No unstable ReLU/Sign neurons left in some domains") and the
    // verdict boundary then DROPS it — every `VnncompResult::Unknown` return site
    // is reasonless, and the eviction/decline telemetry behind it is `info!` under
    // a default `WARN` filter with `RUST_LOG` not honoured. Carry it to the one
    // terminal accounting line instead, so an `unknown` row says WHY it stopped
    // and HOW MUCH of its official budget it left on the table.
    match &result.result {
        BabVerificationStatus::Unknown { reason } => crate::flight::note_stop_reason(reason),
        BabVerificationStatus::Timeout => {
            crate::flight::note_stop_reason("BaB stopped at its deadline (Timeout)")
        }
        _ => {}
    }
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
#[path = "dispatch_graph_ibp_tests.rs"]
mod graph_ibp_tests;

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

    #[test]
    fn postbab_reserve_deadline_marker_requires_a_typed_engine_source() {
        let verification_start = Instant::now()
            .checked_sub(Duration::from_millis(10))
            .expect("ten milliseconds before now is representable");
        let deadline = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond before now is representable");
        let result = |status| BetaCrownResult {
            result: status,
            domains_explored: 0,
            domains_verified: 0,
            cuts_generated: 0,
            max_depth_reached: 0,
            time_elapsed: Duration::ZERO,
            output_bounds: None,
        };

        super::super::output::begin_capture();
        super::super::output::record_captured_deadline_plan(
            Duration::from_millis(10),
            Some(deadline),
            true,
        );
        let carried = gate_result_at_deadline(
            result(BabVerificationStatus::Timeout),
            verification_start,
            Some(deadline),
        );
        let carried_plan = super::super::output::take_captured_deadline_plan()
            .expect("capture records the configured deadline plan");
        super::super::output::end_capture();
        assert!(matches!(carried.result, BabVerificationStatus::Timeout));
        assert!(
            !carried_plan.configured_deadline_caused_terminal_timeout,
            "a pre-existing phase timeout must not be credited to the later configured wall"
        );

        super::super::output::begin_capture();
        super::super::output::record_captured_deadline_plan(
            Duration::from_millis(10),
            Some(deadline),
            true,
        );
        let converted = gate_result_at_deadline(
            result(BabVerificationStatus::Verified),
            verification_start,
            Some(deadline),
        );
        let converted_plan = super::super::output::take_captured_deadline_plan()
            .expect("capture records the configured deadline plan");
        super::super::output::end_capture();
        assert!(matches!(converted.result, BabVerificationStatus::Timeout));
        assert!(
            !converted_plan.configured_deadline_caused_terminal_timeout,
            "late publication suppression does not prove the reserve created verifier work"
        );

        super::super::output::begin_capture();
        super::super::output::record_captured_deadline_plan(
            Duration::from_millis(10),
            Some(deadline),
            true,
        );
        let timeout = result(BabVerificationStatus::Timeout);
        mark_captured_engine_timeout_source(&timeout, Some(deadline));
        let sourced_plan = super::super::output::take_captured_deadline_plan()
            .expect("capture records the configured deadline plan");
        super::super::output::end_capture();
        assert!(
            sourced_plan.configured_deadline_caused_terminal_timeout,
            "an exact typed engine-global timeout source is causal evidence"
        );
    }
}

#[cfg(all(test, feature = "mip"))]
mod tests {
    use super::*;
    use ndarray::arr2;
    use ny_propagate::layers::{LinearLayer, ReLULayer, SigmoidLayer};
    use ny_propagate::{Layer, Network, PhaseBudgetConfig};
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

    /// #safenlp-cycle: the one full-cycle gate is exact and default-dark,
    /// resolved through its ny-levers declaration (no raw env read; env-free
    /// via the injected lookup).
    #[test]
    fn safenlp_mip_cycle_gate_accepts_only_literal_one() {
        let decl = &ny_levers::decls::safenlp_cycle::SAFENLP_MIP_CYCLE;
        assert!(ny_levers::read_with(decl, |_| Some("1".to_string()))
            .value
            .as_bool());
        for raw in [
            None,
            Some("0"),
            Some(" 1"),
            Some("1 "),
            Some("true"),
            Some("01"),
        ] {
            assert!(
                !ny_levers::read_with(decl, |_| raw.map(str::to_string))
                    .value
                    .as_bool(),
                "non-exact cycle spelling {raw:?} must leave every historical path byte-identical"
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
        // #mip-demand: this regression is about the layer-set route, so both
        // new predicates stay at their fail-safe "no static claim" values.
        let routed = apply_bab_mip_budget_policy(
            base,
            CompleteVerifierArg::Auto,
            true,
            false,
            false,
            false,
            None,
        );

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
            // #reserve-demand: this regression is about the MIP reservation, so
            // the post-BaB demand finding stays at its fail-safe `false`.
            false,
            // #mip-demand: and so do the two new static predicates — the
            // SafeNLP shared-prefix treatment must survive all three.
            false,
            None,
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
            false,
            false,
            None,
        );
        assert!(
            !bab_only.mip_reservation_armed_for_test(),
            "explicit complete-verifier=bab must remain no-MIP under the gate"
        );
    }

    /// #mip-demand — the production policy chain must carry the two NEW static
    /// predicates, and the SafeNLP shared-prefix treatment must survive both.
    ///
    /// The exemption argument is the same one `#deadlane` already makes: that
    /// treatment's later MIP runs on a RELOADED SEQUENTIAL net, which neither
    /// the graph encode-nnz probe nor the graph whole-net slice floor describes.
    #[test]
    fn production_budget_dispatch_carries_the_mip_demand_predicates() {
        // ENV-FREE by construction, unlike its two older siblings above: every
        // ledger here forces `with_mip_reservation(true)`, so nothing depends on
        // `graph_mip_enabled()` and the test needs no `NY_GRAPH_MIP` guard. That
        // guard is a raw environment read, and the ny-levers ratchet counts it —
        // adding a fourth one to buy a test precondition is exactly the debt the
        // gate exists to refuse. Forcing the flag is also strictly more robust:
        // the guard could only DETECT a hostile ambient value, not survive it.
        let policy = PhaseBudgetConfig {
            post_bab_pgd_fraction: 0.10,
            ..PhaseBudgetConfig::default()
        };

        // Layer set fine, but the model is statically over the encode-nnz cap.
        let over_cap = apply_bab_mip_budget_policy(
            PhaseBudgetLedger::new(114, policy.clone()).with_mip_reservation(true),
            CompleteVerifierArg::Auto,
            false,
            false,
            false,
            true,
            Some(20),
        );
        assert_eq!(over_cap.mip_statically_ineligible(), "nnz-cap");
        assert!(!over_cap.mip_reservation_armed_for_test());

        // No static claim at all: the historical reservation survives intact.
        let untouched = apply_bab_mip_budget_policy(
            PhaseBudgetLedger::new(114, policy.clone()).with_mip_reservation(true),
            CompleteVerifierArg::Auto,
            false,
            false,
            false,
            false,
            Some(20),
        );
        assert_eq!(
            untouched.mip_statically_ineligible(),
            "none",
            "the chain must be able to produce the OTHER answer"
        );
        assert!(untouched.mip_reservation_armed_for_test());
        let no_floor = apply_bab_mip_budget_policy(
            PhaseBudgetLedger::new(114, policy).with_mip_reservation(true),
            CompleteVerifierArg::Auto,
            false,
            false,
            false,
            false,
            None,
        );
        assert_eq!(
            untouched.planned_mip_timeout_at_bab_deadline(),
            no_floor.planned_mip_timeout_at_bab_deadline(),
            "an ADMISSIBLE reserve (28 s >= the 20 s floor) must be byte-identical \
             with and without the floor declared"
        );

        // SafeNLP shared-prefix: exempt from BOTH new predicates.
        let repaired = apply_bab_mip_budget_policy(
            PhaseBudgetLedger::new(15, safenlp_official_internal_policy())
                .with_mip_reservation(true),
            CompleteVerifierArg::Auto,
            false,
            true,
            false,
            true,
            Some(20),
        );
        assert!(
            repaired.mip_reservation_armed_for_test(),
            "sequential reload reachability must survive the graph nnz probe"
        );
        assert_eq!(
            repaired.mip_statically_ineligible(),
            "none",
            "an exempted ledger must record no static finding"
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
