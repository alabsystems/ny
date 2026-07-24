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
use super::output::output_result;
use super::verify::phase_budget::PhaseBudgetLedger;
use super::verify::{
    confirm_potential_violation, normalize_result_wall_time, try_pgd_before_mip,
    verify_relational_constraints, verify_standard,
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

/// Pure layer-type predicate: is this layer one the HiGHS MIP path can encode
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
            let (out_dim, in_dim) = linear.weight.dim();
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
/// MIP (HiGHS, exact Big-M) is a sound complete verifier, so escalating can
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
    pub(super) gemm_engine: Option<&'a dyn GemmEngine>,
    pub(super) compute_device: Option<Arc<ComputeDevice>>,
    pub(super) allow_heuristic_logsoftmax: bool,
    pub(super) allow_heuristic_softmax: bool,
    pub(super) input_split_metrics_jsonl: Option<&'a Path>,
    pub(super) domain_batch_metrics_jsonl: Option<&'a Path>,
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

/// Run the sequential MIP-only path, including the PGD warm-start precheck.
pub(super) fn run_mip_only(ctx: &DispatchContext<'_>, mip_solver: MipSolverArg) -> Result<()> {
    let mip_start = Instant::now();
    let mip_ledger = PhaseBudgetLedger::new(
        ctx.config.timeout.as_secs(),
        ctx.config.phase_budget.clone(),
    );

    let warm_start_candidate = if ctx.run_upfront_pgd {
        match &*ctx.model_net {
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
                        let result = BetaCrownResult {
                            result: BabVerificationStatus::Violated {
                                counterexample: counterexample.iter().copied().collect(),
                                output: output.iter().copied().collect(),
                            },
                            domains_explored: 0,
                            domains_verified: 0,
                            cuts_generated: 0,
                            max_depth_reached: 0,
                            time_elapsed: mip_start.elapsed(),
                            output_bounds: None,
                        };
                        output_result(
                            &result,
                            ctx.property,
                            ctx.epsilon,
                            ctx.effective_threshold,
                            ctx.verify_upper,
                            ctx.json,
                            ctx.sigmoid_peeled,
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

    let Some(mip_timeout) = mip_ledger
        .remaining()
        .map(|duration| duration.as_secs())
        .filter(|&seconds| seconds > 0)
    else {
        let result = BetaCrownResult {
            result: BabVerificationStatus::Timeout,
            domains_explored: 0,
            domains_verified: 0,
            cuts_generated: 0,
            max_depth_reached: 0,
            time_elapsed: mip_start.elapsed(),
            output_bounds: None,
        };
        output_result(
            &result,
            ctx.property,
            ctx.epsilon,
            ctx.effective_threshold,
            ctx.verify_upper,
            ctx.json,
            ctx.sigmoid_peeled,
        )?;
        return Ok(());
    };

    // Every solver arg — including the default AY backend, which lowers the
    // same ny-mip IR to the external ay solver — routes through the shared
    // verify_with_mip pipeline (encoder, phase-split, witness revalidation).
    mip_highs::verify_with_mip(
        ctx.model_net,
        ctx.input,
        ctx.vnnlib_spec,
        ctx.property.as_deref(),
        ctx.epsilon,
        ctx.effective_threshold,
        mip_timeout,
        warm_start_candidate.as_ref(),
        mip_solver,
        ctx.json,
    )
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
    let _ = (complete_verifier, mip_solver);
    // One ledger owns the overall wall-clock deadline in every build.  MIP
    // builds additionally consume its escalation budget, while all builds use
    // the same deadline to bound optional post-verdict certificate emission.
    let bab_ledger = PhaseBudgetLedger::new(
        ctx.config.timeout.as_secs(),
        ctx.config.phase_budget.clone(),
    );
    let bab_start = Instant::now();

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
                output_result(
                    &result,
                    ctx.property,
                    ctx.epsilon,
                    ctx.effective_threshold,
                    ctx.verify_upper,
                    ctx.json,
                    ctx.sigmoid_peeled,
                )?;
                super::cert_adapter::maybe_emit_certificate(
                    ctx,
                    &result,
                    bab_ledger.overall_deadline(),
                );
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
            bab_start + ctx.config.timeout,
        ) {
            // A `Violated` result carries a concrete, in-process-confirmed
            // witness; the vnncomp harness re-confirms it against the
            // ONNX-Runtime trusted oracle before any `sat` is scored.
            output_result(
                &result,
                ctx.property,
                ctx.epsilon,
                ctx.effective_threshold,
                ctx.verify_upper,
                ctx.json,
                ctx.sigmoid_peeled,
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
            bab_start + ctx.config.timeout,
        ) {
            output_result(
                &result,
                ctx.property,
                ctx.epsilon,
                ctx.effective_threshold,
                ctx.verify_upper,
                ctx.json,
                ctx.sigmoid_peeled,
            )?;
            return Ok(());
        }
    }

    let mut verifier = match ctx.compute_device.clone() {
        Some(device) => BetaCrownVerifier::new_with_engine(ctx.config.clone(), device),
        None => BetaCrownVerifier::new(ctx.config.clone()),
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
        graph.set_use_patches_mode(ctx.config.use_patches());
    }

    // Graph-MIP LEAF oracle (increment 6, `docs/GRAPH_MIP_LEAF_SOLVER.md`):
    // the graph ReLU-split BaB decides stuck deep subdomains exactly (split
    // premises pinned, certified-UNSAT admission) instead of requeueing them.
    // DEFAULT-ON (2026-07-21, sound + time-sliced); `NY_GRAPH_MIP_LEAF=0`
    // detaches the oracle ⇒ every BaB path byte-identical (the old default).
    #[cfg(feature = "mip")]
    if let Some(oracle) =
        super::graph_mip_leaf::maybe_graph_mip_leaf_oracle(mip_solver.mip_backend())
    {
        verifier = verifier.with_graph_mip_leaf_oracle(oracle);
    }

    // Single-clause per-clause-box disjunctions (nn4sys mscn `_dual`
    // cardinality_1_1: one `(and <input box> <band constraint>)` disjunct)
    // parse to ONE output constraint, so the plain `len() > 1` gate used to
    // send them to `verify_standard` — whose global-box f32 BaB can never
    // decide a ±1e-5 band. Route them through the relational dispatch, which
    // forwards per-clause-box shapes to the box-refinement screen (+ sound
    // f64 leaf escalation). Verdict semantics are identical for one clause.
    let has_multi_constraints = ctx.vnnlib_spec.is_some_and(|vnnlib| {
        vnnlib.output_constraints.len() > 1
            || ctx.has_relational
            || vnnlib.has_boxed_clause_disjunction()
    });

    let result = if let (true, Some(vnnlib)) = (has_multi_constraints, ctx.vnnlib_spec) {
        verify_relational_constraints(
            &*ctx.model_net,
            ctx.input,
            vnnlib,
            ctx.config,
            &verifier,
            ctx.use_relu_split,
            ctx.gpu_bab,
            ctx.run_upfront_pgd,
            ctx.config.pgd_restarts,
            ctx.config.pgd_steps,
            ctx.config.timeout.as_secs(),
            ctx.gemm_engine,
            ctx.json,
        )?
    } else {
        if !ctx.json {
            info!("Running β-CROWN...");
        }
        // Thread the wall-clock deadline (#4321) so the initial-bound pass and
        // BaB phase budgets are measured from `bab_start` — which already
        // accounts for time spent in model setup and the IBP fast-path above.
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
            Some(bab_start + ctx.config.timeout),
        )?
    };

    let deadline = Some(bab_start + ctx.config.timeout);
    let result = confirm_potential_violation(
        &*ctx.model_net,
        ctx.input,
        ctx.vnnlib_spec,
        result,
        ctx.config,
        deadline,
        ctx.json,
    )?;
    let result = normalize_result_wall_time(result, bab_start);

    // Auto-escalate to the MIP complete verifier when BaB is inconclusive
    // (#4246). Under `--features mip` this fires for both an explicit
    // `--complete-verifier mip` (unconditionally, preserving the prior
    // behavior) and the default `auto` policy (gated on MIP-encodability so we
    // only escalate when HiGHS can actually encode the net within the size
    // cap). Explicit `--complete-verifier bab` never escalates. Escalation is
    // sound: HiGHS Big-M MIP is exact, so it can only turn unknown/timeout into
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
            let mip_timeout = bab_ledger.mip_timeout().unwrap_or(0);
            let remaining_from_budget = bab_ledger.remaining_secs_clamped();
            info!(
                "MIP escalation gate: policy_allows={policy_allows} bab_status={:?} mip_timeout={mip_timeout}s remaining={remaining_from_budget}s graph_mip_enabled={}",
                result.result,
                super::graph_mip::graph_mip_enabled()
            );
            if mip_timeout >= 5 {
                // Reload a fresh sequential network for MIP encoding (the BaB
                // model may be a graph). Check encodability on the network we
                // would actually hand to HiGHS.
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
                    if let (BetaCrownModel::Graph(graph), Some(vnnlib)) =
                        (&*ctx.model_net, ctx.vnnlib_spec)
                    {
                        match super::graph_mip_escalate::try_graph_mip_escalation(
                            graph,
                            ctx.input,
                            vnnlib,
                            &verifier,
                            complete_verifier,
                            &result.result,
                            mip_timeout,
                            mip_solver,
                            ctx.property.as_deref(),
                            ctx.epsilon,
                            ctx.effective_threshold,
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
                    if should_auto_escalate_to_mip(complete_verifier, &result.result, encodable) {
                        let escalation_kind = if complete_verifier == CompleteVerifierArg::Auto {
                            "auto-escalating"
                        } else {
                            "falling back"
                        };
                        info!(
                        "BaB inconclusive ({:?}), {} to MIP with {}s budget ({}s remaining from BaB)",
                        result.result, escalation_kind, mip_timeout, remaining_from_budget
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
                            ctx.epsilon,
                            ctx.effective_threshold,
                            mip_timeout,
                            None,
                            mip_solver,
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
    output_result(
        &result,
        ctx.property,
        ctx.epsilon,
        ctx.effective_threshold,
        ctx.verify_upper,
        ctx.json,
        ctx.sigmoid_peeled,
    )?;
    super::cert_adapter::maybe_emit_certificate(ctx, &result, bab_ledger.overall_deadline());

    Ok(())
}

#[cfg(all(test, feature = "mip"))]
mod tests {
    use super::*;
    use ndarray::arr2;
    use ny_propagate::layers::{LinearLayer, ReLULayer, SigmoidLayer};
    use ny_propagate::{Layer, Network};

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
        // A Sigmoid layer is NOT encodable by the Big-M HiGHS path: escalating
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
        // we keep the BaB unknown/timeout rather than bail in HiGHS.
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
            BabVerificationStatus::PotentialViolation,
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

    #[cfg(feature = "mip")]
    #[test]
    fn strict_graph_mip_decline_has_no_legacy_encoder_fallback() {
        // The strict path owns the NNZ admission decision. Keep this source
        // guard so a future cleanup cannot accidentally restore the old
        // second call after a 5M-NNZ decline and bypass the memory envelope.
        let dispatch = include_str!("dispatch.rs");
        let strict_call = ["super::graph_mip_escalate::", "try_graph_mip_escalation("].concat();
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
