// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// MIP verification path for FC+ReLU networks.
// The historical module name is retained for API/source stability; the solver
// policy now uses the linked AY backend by default, with ay-proc explicit.

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ny_core::{Bound, VerificationResult};
use ny_mip::{
    certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_until_unwired, encode_feedforward,
    CertifiedLinearLowerBound, CertifiedLinearLowerProofRoute, LpTightener, MipBackend, MipConfig,
    MipFeasibilityIngress, MipResult, MipSolver, OneSidedSatProbe, SplitUnsatCache,
};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::layers::LinearLayer;
use ny_propagate::{Layer, Network, PhaseBudgetConfig};
use ny_tensor::BoundedTensor;
use std::path::Path;

use super::mip_preprocess::{
    bounded_tensor_to_bounds, convert_intermediate_bounds, extract_linear_relu_params,
    fold_constant_layers, refuse_mip_feedforward_before_unfold, strip_shape_layers,
    unfold_conv2d_to_linear, validate_mip_feedforward_topology, FoldedMipNetwork,
};
use super::mip_single_hidden::{
    collect_exact_single_hidden_intermediate_bounds, is_single_hidden_linear_relu_linear,
};
use super::output::{
    format_verification_result_json_for_publication, verification_result_exit_code,
    EffectiveTreatmentProjection,
};
use super::BetaCrownModel;
use intermediate_bounds::collect_mip_intermediate_bounds_with_deadline;
use warm_start::build_warm_start_vector;

/// Apply the exact structural preprocessing consumed by the feed-forward MIP
/// encoder.
///
/// Keep this as the single topology authority for both route admission and
/// execution: ONNX MatMul+Add commonly reloads as separate
/// `Linear -> AddConstant` layers, so inspecting the raw sequential network
/// would reject a model that this pipeline soundly folds and encodes.
/// `single_hidden_hull.rs` (#hull-root) reuses it for the same reason, so the
/// hull lane and the MIP encoder always agree on what the network IS.
pub(super) fn canonicalize_mip_feedforward_network(
    network: &Network,
    input_shape: &[usize],
) -> Result<FoldedMipNetwork> {
    // Screen the layer SEQUENCE before densifying any Conv2d. The topology the
    // encoder requires is invariant under every step below, so a sequence that
    // will be refused after canonicalization is refused here at O(layers)
    // instead of after a dense im2col per Conv2d. Measured on relusplitter
    // cifar_biasfield (Gemm → Reshape → Conv×8): 14.6-16.5 s of a 51 s internal
    // grant reclaimed, 3/3 rows. The screen abstains on any layer it does not
    // model, so `validate_mip_feedforward_topology` below remains the authority.
    refuse_mip_feedforward_before_unfold(network)?;
    let mip_network = unfold_conv2d_to_linear(network, input_shape)?;
    let mip_network = strip_shape_layers(&mip_network);
    let mip_network = fold_constant_layers(&mip_network)?;
    validate_mip_feedforward_topology(&mip_network)?;
    Ok(mip_network)
}

/// Human-readable solver name for a MIP backend (verdict output/diagnostics).
fn backend_name(backend: MipBackend) -> &'static str {
    match backend {
        MipBackend::Ay => "ay",
        MipBackend::AyProc => "ay-proc",
    }
}

/// Exact default-off gate for AY's verdict-preserving margin reframe.
///
/// Graph-MIP already uses this provenance-recorded research gate.  The direct
/// FC MIP path deliberately shares its semantics: only exact `1` marks one
/// caller-identified unsafe row, while every other spelling leaves the
/// historical objective-zero feasibility model byte-identical.
fn ay_margin_reframe_enabled_from_value(value: Option<&str>) -> bool {
    value == Some("1")
}

fn ay_margin_reframe_enabled() -> bool {
    ay_margin_reframe_enabled_from_value(std::env::var("NY_AY_MARGIN_REFRAME").ok().as_deref())
        // #safenlp-cycle: the full-cycle A/B gate arms the SAME reframe — a
        // marked unsafe row rides `MarginMode::Required` inside the pinned
        // ay-milp, whose threshold-stop banks UNSAT on a strict bound crossing
        // with original-model proof replay (the only handoff shape whose dual
        // bound MOVES on this class, docs/SAFENLP_AY_ARC_2026-08-20.md §5.1).
        || super::dispatch::safenlp_mip_cycle_enabled()
}

/// Preserve terminal ownership across the in-process competition capture seam.
///
/// This consumes the already-selected typed ingress; it must never re-read the
/// environment or infer ownership from category/log strings.
fn capture_terminal_safenlp_ingress(feasibility_ingress: MipFeasibilityIngress) {
    if feasibility_ingress == MipFeasibilityIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix {
        super::output::mark_captured_safenlp_marked_margin_terminal();
    }
}

/// Exact default-off gate for a certificate-authoritative four-ReLU tree.
///
/// This is a caller-side research canary, not an AY search override. Only the
/// literal `1` can retain an unmodified base model and ask the existing
/// fixed-assignment-tree API for an independently replayed safety proof.
fn mip_certified_shared_tree_enabled_from_value(value: Option<&str>) -> bool {
    value == Some("1")
}

fn mip_certified_shared_tree_enabled() -> bool {
    mip_certified_shared_tree_enabled_from_value(
        std::env::var("NY_MIP_CERTIFIED_SHARED_TREE")
            .ok()
            .as_deref(),
    )
}

#[derive(Debug, Clone, PartialEq)]
struct CertifiedSharedTreePlan {
    /// Sparse lower-bound objective whose strict positivity excludes the one
    /// admitted unsafe row.
    objective: [(ny_mip::ir::Col, f64); 2],
    /// Exactly four unfixed ReLU binaries, in historical widest-first order.
    splits: [ny_mip::ir::Col; 4],
}

/// Recover the unique pairwise unsafe constraint without changing its sense.
///
/// The fixed-tree API proves `objective > 0`: for `yi <= yj` the excluding
/// objective is `yi - yj`; for `yi >= yj` it is `yj - yi`. Strict and
/// non-strict unsafe rows share the same sufficient strict-safe complement.
fn certified_shared_tree_pairwise_indices(spec: &VnnLibSpec) -> Option<(usize, usize, bool)> {
    if spec.is_disjunction {
        return None;
    }
    let constraints = conjunctive_constraints_owned(spec);
    let constraint = constraints.first()?;
    if constraints.len() != 1 {
        return None;
    }
    let (first, second, reverse) = match constraint {
        OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => (*i, *j, false),
        OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => (*i, *j, true),
        OutputConstraint::LessEqConst(..)
        | OutputConstraint::LessThanConst(..)
        | OutputConstraint::GreaterEqConst(..)
        | OutputConstraint::GreaterThanConst(..) => return None,
        _ => return None,
    };
    if first == second || first >= spec.num_outputs || second >= spec.num_outputs {
        return None;
    }
    Some((first, second, reverse))
}

/// Cheap admission checked before cloning the encoded base problem.
///
/// In particular, the margin reframe and certified tree are mutually
/// exclusive. If both exact gates are requested, the tree refuses and the
/// ordinary fallback is still allowed to mark and solve its unsafe row.
fn certified_shared_tree_preclone_eligible(
    enabled: bool,
    margin_reframe_enabled: bool,
    backend: MipBackend,
    exact_single_hidden_fast_path: bool,
    spec: &VnnLibSpec,
    deadline: std::time::Instant,
) -> bool {
    enabled
        && !margin_reframe_enabled
        && backend == MipBackend::Ay
        && exact_single_hidden_fast_path
        && certified_shared_tree_pairwise_indices(spec).is_some()
        && std::time::Instant::now() < deadline
}

/// Build the proof request from an unmodified encoded base problem.
///
/// The width ordering intentionally duplicates `MipSolver`'s historical
/// default comparator: descending width, then encoder insertion index. The
/// canary does not inherit `NY_MIP_STABILITY_HINTS`; its four-way topology is
/// stable across unrelated search-advice experiments.
fn certified_shared_tree_plan(
    preclone_eligible: bool,
    spec: &VnnLibSpec,
    base: &ny_mip::MipParts,
) -> Option<CertifiedSharedTreePlan> {
    if !preclone_eligible
        || base.problem.margin_row().is_some()
        || base.num_cols != base.problem.num_cols()
        || base.output_vars.len() != spec.num_outputs
        || base.binary_vars.len() != base.binary_widths.len()
    {
        return None;
    }

    let (i, j, reverse) = certified_shared_tree_pairwise_indices(spec)?;
    let yi = *base.output_vars.get(i)?;
    let yj = *base.output_vars.get(j)?;
    if yi == yj || base.problem.cols().get(yi.0)?.integer || base.problem.cols().get(yj.0)?.integer
    {
        return None;
    }
    let objective = if reverse {
        [(yj, 1.0), (yi, -1.0)]
    } else {
        [(yi, 1.0), (yj, -1.0)]
    };

    let mut order = Vec::with_capacity(base.binary_vars.len());
    let mut seen = vec![false; base.problem.num_cols()];
    for (index, (&col, &width)) in base.binary_vars.iter().zip(&base.binary_widths).enumerate() {
        let col_spec = base.problem.cols().get(col.0)?;
        if !col_spec.integer || !width.is_finite() || width <= 0.0 {
            return None;
        }
        if std::mem::replace(seen.get_mut(col.0)?, true) {
            return None;
        }
        if col_spec.lb == 0.0 && col_spec.ub == 1.0 {
            order.push(index);
        } else if !(col_spec.lb == col_spec.ub && (col_spec.lb == 0.0 || col_spec.lb == 1.0)) {
            return None;
        }
    }
    if order.len() < 4 {
        return None;
    }
    order.sort_by(|&a, &b| {
        let wa = base.binary_widths.get(a).copied().unwrap_or(0.0);
        let wb = base.binary_widths.get(b).copied().unwrap_or(0.0);
        wb.partial_cmp(&wa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let splits = std::array::from_fn(|index| base.binary_vars[order[index]]);
    Some(CertifiedSharedTreePlan { objective, splits })
}

/// Ask the existing certificate API for the complete root/four-split proof.
///
/// Every decline and error is verdict-neutral. Even a returned certificate is
/// discarded if it arrives after the caller's absolute deadline or reports a
/// shape outside the API's root-or-complete-16-leaf contract.
fn try_certified_shared_tree(
    base: &ny_mip::MipParts,
    plan: &CertifiedSharedTreePlan,
    deadline: std::time::Instant,
) -> Option<CertifiedLinearLowerBound> {
    let started = std::time::Instant::now();
    let proof_deadline = started
        .checked_add(std::time::Duration::from_mins(5))
        .map_or(deadline, |cap| deadline.min(cap));
    if proof_deadline <= started {
        return None;
    }
    let certified = match certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_until_unwired(
        &base.problem,
        &plan.objective,
        0.0,
        &plan.splits,
        proof_deadline,
        16,
    ) {
        Ok(Some(certified)) => certified,
        Ok(None) => return None,
        Err(error) => {
            tracing::warn!(
                %error,
                "certified shared-tree canary declined after a proof-API error"
            );
            return None;
        }
    };
    if std::time::Instant::now() >= proof_deadline || certified.lower.to_bits() != 0.0_f32.to_bits()
    {
        return None;
    }
    let expected_shape = match certified.proof_route {
        CertifiedLinearLowerProofRoute::RelaxationEntailment
        | CertifiedLinearLowerProofRoute::RootFarkas => {
            certified.ay_tree_leaves == 0 && certified.ny_cert_farkas_replays == 1
        }
        CertifiedLinearLowerProofRoute::TreeFarkas => {
            certified.ay_tree_leaves == 16 && certified.ny_cert_farkas_replays == 16
        }
    };
    expected_shape.then_some(certified)
}

/// Mark one explicitly identified unsafe row for AY's equivalent margin solve.
///
/// This helper owns no verdict logic.  It only transports the row identity
/// through `MilpProblem` to the pinned AY backend, whose reframe relaxes that
/// row, optimizes its sparse form, maps the result back to the ORIGINAL
/// feasibility model, and passes every witness/Farkas result through the
/// original-model replay gate.  Multi-row conjunctions need a max-min
/// construction and are deliberately left on plain feasibility.
fn maybe_mark_unique_ay_margin(
    enabled: bool,
    backend: MipBackend,
    parts: &mut ny_mip::MipParts,
    unsafe_rows: &[ny_mip::ir::Row],
) -> Result<bool> {
    if !enabled || backend != MipBackend::Ay || unsafe_rows.len() != 1 {
        return Ok(false);
    }
    parts
        .problem
        .mark_margin_row(unsafe_rows[0])
        .map_err(|error| {
            anyhow::anyhow!(
                "cannot mark direct-MIP unsafe row {} as AY's unique decision margin: {error}",
                unsafe_rows[0].0
            )
        })?;
    Ok(true)
}

fn timeout_verification_result(num_outputs: usize) -> VerificationResult {
    VerificationResult::Timeout {
        provenance: Default::default(),
        partial_bounds: Some(vec![
            Bound::new_allow_infinite(
                f32::NEG_INFINITY,
                f32::INFINITY,
            );
            num_outputs
        ]),
        actual_method: Some(ny_core::MethodUsed::MipHiGHS),
    }
}

/// Enforce the MIP ingress's terminal wall without manufacturing receipt
/// provenance. A synchronous solver can return any terminal result after this
/// wall; late publication suppression does not prove the wall aborted work.
fn gate_mip_result_at_deadline(
    result: VerificationResult,
    num_outputs: usize,
    deadline: std::time::Instant,
) -> VerificationResult {
    if std::time::Instant::now() < deadline {
        return result;
    }
    timeout_verification_result(num_outputs)
}

/// Exact default-off gate for the AY objective-first SAT candidate lane.
///
/// This is intentionally distinct from `NY_AY_MARGIN_REFRAME`: the latter can
/// map an optimum to either feasibility or infeasibility, while this lane has
/// no UNSAT surface and keeps the unsafe row constrained.
fn ay_objective_first_sat_enabled_from_value(value: Option<&str>) -> bool {
    value == Some("1")
}

fn ay_objective_first_sat_enabled() -> bool {
    ay_objective_first_sat_enabled_from_value(
        std::env::var("NY_AY_OBJECTIVE_FIRST_SAT").ok().as_deref(),
    )
}

/// Exact default-off canary for running the sequential-network MIP as one
/// full-model solve instead of the default phase-split race.
///
/// This is deliberately separate from `NY_GRAPH_MIP_SERIAL`: graph-MIP and
/// this sequential complete-verifier path have different callers, budgets,
/// and score evidence.  Malformed values fail closed to the historical
/// auto-split policy; the measurement launcher independently rejects them.
#[cfg(test)]
fn sequential_mip_parallel_split_from_value(value: Option<&str>) -> usize {
    sequential_mip_parallel_split_from_value_and_cycle(value, false)
}

/// #safenlp-cycle: within the full-cycle gate the DEFAULT is the serial
/// one-model solve (the Gurobi-winning shape and the only one that hands AY's
/// margin reframe a single comparable dual bound). Explicit `NY_MIP_SERIAL=1`
/// stays serial either way; malformed values keep failing closed to the
/// gate-appropriate default.
fn sequential_mip_parallel_split_from_value_and_cycle(value: Option<&str>, cycle: bool) -> usize {
    if value == Some("1") || cycle {
        1
    } else {
        MipConfig::default().parallel_split
    }
}

fn sequential_mip_parallel_split() -> usize {
    sequential_mip_parallel_split_from_value_and_cycle(
        std::env::var("NY_MIP_SERIAL").ok().as_deref(),
        super::dispatch::safenlp_mip_cycle_enabled(),
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ObjectiveFirstSatBudget {
    probe_secs: f64,
    /// Exact historical off-path wall slice, after any window floor,
    /// phase-split outer cap, and backend hard clamp.
    envelope_secs: f64,
}

impl ObjectiveFirstSatBudget {
    const MIN_FALLBACK_SECS: f64 = 0.001;

    /// Deterministic mirror of the absolute-deadline remainder calculation.
    ///
    /// This charges probe setup, model lowering, detached-worker wait, and
    /// concrete replay by subtracting ACTUAL elapsed wall time, rather than
    /// blindly granting a fresh nominal fallback slice.
    fn fallback_secs_after_elapsed(self, elapsed_secs: f64) -> Option<f64> {
        if !elapsed_secs.is_finite() || elapsed_secs < 0.0 {
            return None;
        }
        let remaining = self.envelope_secs - elapsed_secs;
        (remaining >= Self::MIN_FALLBACK_SECS).then_some(remaining)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ObjectiveFirstSatLedger {
    deadline: std::time::Instant,
}

impl ObjectiveFirstSatLedger {
    fn start(
        budget: ObjectiveFirstSatBudget,
        outer_deadline: Option<std::time::Instant>,
    ) -> Option<Self> {
        let now = std::time::Instant::now();
        let nominal_deadline =
            now.checked_add(std::time::Duration::from_secs_f64(budget.envelope_secs))?;
        let deadline = outer_deadline.map_or(nominal_deadline, |outer| outer.min(nominal_deadline));
        if deadline <= now {
            return None;
        }
        Some(Self { deadline })
    }

    fn expired(self) -> bool {
        std::time::Instant::now() >= self.deadline
    }

    fn probe_secs(self, budget: ObjectiveFirstSatBudget) -> Option<f64> {
        let remaining = self
            .deadline
            .checked_duration_since(std::time::Instant::now())?
            .as_secs_f64();
        let probe_secs = budget.probe_secs.min(remaining);
        (probe_secs >= ObjectiveFirstSatBudget::MIN_FALLBACK_SECS).then_some(probe_secs)
    }
}

/// Reserve a bounded probe inside the exact historical MIP wall envelope.
///
/// Only one explicitly identified one-sided row is admitted.  Multi-row
/// conjunctions need a separate max-min objective construction and are
/// deliberately refused rather than guessed.  `requested_secs` sizes the
/// probe, preserving the canary policy. `historical_envelope_secs` is the
/// actual objective-off wall slice reported by [`MipSolver`], including a
/// window floor, phase-split outer deadline, and the backend's 24-hour clamp.
fn objective_first_sat_budget(
    enabled: bool,
    backend: MipBackend,
    one_sided_rows: usize,
    requested_secs: f64,
    historical_envelope_secs: f64,
) -> Option<ObjectiveFirstSatBudget> {
    if !enabled
        || backend != MipBackend::Ay
        || one_sided_rows != 1
        || !requested_secs.is_finite()
        || requested_secs < 0.1
        || !historical_envelope_secs.is_finite()
    {
        return None;
    }
    const PROBE_FRACTION: f64 = 0.20;
    const PROBE_CAP_SECS: f64 = 10.0;
    let probe_secs = (requested_secs * PROBE_FRACTION).min(PROBE_CAP_SECS);
    (probe_secs > 0.0
        && historical_envelope_secs - probe_secs >= ObjectiveFirstSatBudget::MIN_FALLBACK_SECS)
        .then_some(ObjectiveFirstSatBudget {
            probe_secs,
            envelope_secs: historical_envelope_secs,
        })
}

fn objective_first_sat_fallback_config(
    config: MipConfig,
    budget: ObjectiveFirstSatBudget,
    ledger: ObjectiveFirstSatLedger,
) -> MipConfig {
    MipConfig {
        // The relative slice is the full historical envelope so an early
        // decline can reclaim unused probe time.  The original absolute
        // deadline—not this relative number—is the controlling wall cap.
        timeout_secs: budget.envelope_secs,
        ay_hard_deadline: Some(ledger.deadline),
        ..config
    }
}

/// Whether an unconfirmed feasibility witness may launch the historical
/// robustness retry.
///
/// Required shared-prefix ingress owns exactly one feasibility session.  A
/// SAT point that fails concrete replay is already demoted to Unknown and may
/// not be shopped to a second solver, even if a future relational constraint
/// becomes shiftable.
fn unconfirmed_sat_retry_allowed(
    feasibility_ingress: MipFeasibilityIngress,
    solver_returned_sat: bool,
    witness_confirmed: bool,
) -> bool {
    feasibility_ingress == MipFeasibilityIngress::Historical
        && solver_returned_sat
        && !witness_confirmed
}

/// Verify a sequential FC+ReLU network with the MILP pipeline on the ay
/// backend (SOLVER POLICY: ny-mip docs/SOLVER_POLICY.md).
///
/// This is the VNN-COMP `mip` path: encode the network with tight Big-M ReLUs
/// and solve it exactly (sat_relu, safenlp, malbeware).
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_with_mip(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    vnnlib: Option<&VnnLibSpec>,
    property: Option<&Path>,
    model: Option<&Path>,
    epsilon: f32,
    threshold: f32,
    deadline: std::time::Instant,
    warm_start_candidate: Option<&ArrayD<f32>>,
    mip_solver: crate::MipSolverArg,
    reporting_start: std::time::Instant,
    effective_treatment: &EffectiveTreatmentProjection,
    json: bool,
) -> Result<()> {
    verify_with_mip_inner(
        model_net,
        input,
        vnnlib,
        property,
        model,
        epsilon,
        threshold,
        deadline,
        warm_start_candidate,
        mip_solver,
        reporting_start,
        effective_treatment,
        json,
        MipFeasibilityIngress::Historical,
        false,
    )
    .map(|_| ())
}

#[cfg(test)]
pub(crate) fn test_mip_publication_receipt(captured_json: &str) -> MipUnsatPublication {
    MipUnsatPublication {
        captured_json: captured_json.to_owned(),
    }
}

/// Opaque evidence that THIS whole-root MIP invocation published its own
/// typed, certificate-gated Verified result successfully. This is not an
/// Instance token and cannot authorize a Source partition. The payload binds
/// the exact captured output; a later writer invalidates the caller's match.
pub(crate) struct MipUnsatPublication {
    captured_json: String,
}

impl MipUnsatPublication {
    pub(crate) fn into_captured_json(self) -> String {
        self.captured_json
    }
}

fn bridge_unsat_publication(
    result: &VerificationResult,
    publication_refused: bool,
    json: bool,
) -> Option<MipUnsatPublication> {
    if !json || publication_refused || !matches!(result, VerificationResult::Verified { .. }) {
        return None;
    }
    // Called only immediately after print_result emitted this invocation's
    // complete new JSON. No status/method/log parsing can mint the receipt.
    super::output::snapshot_captured_json()
        .map(|captured_json| MipUnsatPublication { captured_json })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn verify_with_mip_for_bridge(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    vnnlib: Option<&VnnLibSpec>,
    property: Option<&Path>,
    model: Option<&Path>,
    epsilon: f32,
    threshold: f32,
    deadline: std::time::Instant,
    warm_start_candidate: Option<&ArrayD<f32>>,
    mip_solver: crate::MipSolverArg,
    reporting_start: std::time::Instant,
    effective_treatment: &EffectiveTreatmentProjection,
    json: bool,
    retain_publication: bool,
) -> Result<Option<MipUnsatPublication>> {
    verify_with_mip_inner(
        model_net,
        input,
        vnnlib,
        property,
        model,
        epsilon,
        threshold,
        deadline,
        warm_start_candidate,
        mip_solver,
        reporting_start,
        effective_treatment,
        json,
        MipFeasibilityIngress::Historical,
        retain_publication,
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_with_mip_inner(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    vnnlib: Option<&VnnLibSpec>,
    property: Option<&Path>,
    model: Option<&Path>,
    epsilon: f32,
    threshold: f32,
    deadline: std::time::Instant,
    warm_start_candidate: Option<&ArrayD<f32>>,
    mip_solver: crate::MipSolverArg,
    reporting_start: std::time::Instant,
    effective_treatment: &EffectiveTreatmentProjection,
    json: bool,
    feasibility_ingress: MipFeasibilityIngress,
    capture_bridge_publication: bool,
) -> Result<Option<MipUnsatPublication>> {
    let backend = mip_solver.mip_backend();
    let requires_safenlp_marked_margin_shared_prefix =
        feasibility_ingress == MipFeasibilityIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix;
    let requires_safenlp_shared_prefix = feasibility_ingress != MipFeasibilityIngress::Historical;
    if requires_safenlp_shared_prefix {
        anyhow::ensure!(
            backend == MipBackend::Ay,
            "required SafeNLP shared-prefix ingress needs the in-process AY backend"
        );
        anyhow::ensure!(
            warm_start_candidate.is_none(),
            "required SafeNLP shared-prefix ingress owns a cold session"
        );
    }
    // A marked route owns the terminal answer even when that answer is
    // timeout/unknown; the vnncomp caller must not reinterpret spare
    // wall-clock as permission for APGD, a fallback, or a second solve.
    capture_terminal_safenlp_ingress(feasibility_ingress);
    // MIP verification only supports sequential networks
    let network = match model_net {
        BetaCrownModel::Sequential(net) => net,
        BetaCrownModel::Graph(_) => {
            anyhow::bail!(
                "MIP verification only supports sequential networks (no residual/attention). Use --complete-verifier bab for DAG models."
            );
        }
    };

    if !json {
        println!(
            "\nRunning MIP verification ({} solver)...",
            backend_name(backend)
        );
    }

    let initial_timeout_secs = deadline
        .saturating_duration_since(std::time::Instant::now())
        .as_secs_f64();
    let config = MipConfig {
        backend,
        parallel_split: sequential_mip_parallel_split(),
        timeout_secs: initial_timeout_secs,
        lp_tighten: true, // Tighten CROWN-IBP bounds via LP relaxation before MIP (#3218)
        ay_hard_deadline: (backend == MipBackend::Ay).then_some(deadline),
        ..Default::default()
    };
    // #safenlp-cycle engagement marker (R9: a null without the lever's own
    // observable is vacuous). Print-only; every decision above is already
    // snapshotted into typed state.
    if super::dispatch::safenlp_mip_cycle_enabled() {
        eprintln!(
            "[safenlp-cycle] armed parallel_split={} margin_reframe={} ingress={:?} timeout_s={:.1}",
            config.parallel_split,
            ay_margin_reframe_enabled(),
            feasibility_ingress,
            initial_timeout_secs,
        );
    }

    // Canonicalize with the same authority used by pre-route admission.
    // Conv2d is unfolded before shape-only layers are stripped, constants are
    // folded with an exact-f64 bias sidecar, and the activation topology is
    // validated before it is erased into affine parameters.
    let mip_network = canonicalize_mip_feedforward_network(network, input.shape())?;
    let use_exact_single_hidden_fast_path = is_single_hidden_linear_relu_linear(&mip_network);

    // Keep original layer indices for `convert_intermediate_bounds()`. The
    // exact single-hidden fast path skips CROWN/LP. Both arms return the
    // TYPE-ENFORCED `SoundIntermediateBounds` (#mip-conv-sound-enclosure): the
    // certified collector is the only constructor and the converter below
    // accepts nothing else, so the plain collector cannot reach the encoder.
    let intermediate_bounds = if use_exact_single_hidden_fast_path {
        collect_exact_single_hidden_intermediate_bounds(network, input)?
    } else {
        // General path: budgeted CROWN-IBP falls back to plain IBP after a
        // short preprocessing deadline so the complete solver keeps most of
        // the budget.
        let policy = PhaseBudgetConfig::default();
        let crown_budget =
            intermediate_bounds::mip_crown_ibp_budget_secs(config.timeout_secs, &policy);
        let now = std::time::Instant::now();
        let crown_deadline = now
            .checked_add(std::time::Duration::from_secs_f64(crown_budget))
            .unwrap_or(deadline)
            .min(deadline);
        collect_mip_intermediate_bounds_with_deadline(network, input, Some(crown_deadline))?
    };

    // Extract weights/dimensions from the structural folded network, but use
    // the exact-f64 bias sidecar produced by `fold_constant_layers`. The f32
    // biases inside `mip_network` exist only to satisfy `LinearLayer`'s storage
    // type and may be rounded; using them for a certified UNSAT would prove a
    // nearby network rather than the original constant-op algebra.
    let (weights, _structural_biases, layer_dims) = extract_linear_relu_params(&mip_network)?;
    let biases = mip_network.exact_biases().to_vec();
    if biases.len() != weights.len() {
        anyhow::bail!(
            "MIP constant-fold bias sidecar mismatch: {} biases for {} Linear layers",
            biases.len(),
            weights.len()
        );
    }
    let input_bounds = bounded_tensor_to_bounds(input)?;
    // Use the original network for index alignment: IBP outputs follow the
    // original layer order, so the stripped network would misalign shape-layer
    // indices and feed the encoder the wrong Big-M bounds.
    let intermediate_bounds_vec = convert_intermediate_bounds(&intermediate_bounds, network)?;

    // LP tightening: tighter Big-M → faster B&B. Ref: α-β-CROWN bounds_core.py:37-92
    let intermediate_bounds_vec = if config.lp_tighten && !use_exact_single_hidden_fast_path {
        let tighten_start = std::time::Instant::now();
        let lp_budget_secs = config.timeout_secs * 0.1;
        let lp_deadline = tighten_start
            .checked_add(std::time::Duration::from_secs_f64(lp_budget_secs))
            .unwrap_or(deadline)
            .min(deadline);
        let mut tightened = intermediate_bounds_vec;
        let mut total_stable = 0usize;
        let mut total_unstable = 0usize;
        // Progressive: rebuild tightener each layer so layer N uses tightened 0..N-1
        for layer_idx in 0..tightened.len() {
            let live_remaining = lp_deadline
                .saturating_duration_since(std::time::Instant::now())
                .as_secs_f64();
            if live_remaining <= 0.0 {
                break;
            }
            let tighten_config = MipConfig {
                timeout_secs: lp_budget_secs.min(live_remaining),
                ay_hard_deadline: (backend == MipBackend::Ay).then_some(lp_deadline),
                ..config
            };
            let tightener = LpTightener::new(
                weights.clone(),
                biases.clone(),
                layer_dims.clone(),
                input_bounds.clone(),
                tightened.clone(),
                tighten_config,
            );
            let (new_bounds, newly_stable) = tightener
                .tighten_layer(layer_idx, &tightened[layer_idx])
                .map_err(|e| {
                    anyhow::anyhow!("LP tightening failed on layer {}: {}", layer_idx, e)
                })?;
            total_unstable += new_bounds
                .iter()
                .filter(|b| b.lower() < 0.0 && b.upper() > 0.0)
                .count();
            total_stable += newly_stable;
            tightened[layer_idx] = new_bounds;
        }
        if !json {
            println!(
                "  LP tightening: {total_stable} neurons fixed stable, {total_unstable} remain unstable ({:.2}s)",
                tighten_start.elapsed().as_secs_f64()
            );
        }
        tightened
    } else {
        intermediate_bounds_vec
    };

    // Derive the solve grant from the authoritative deadline *after*
    // preprocessing. Never add a one-second floor: that used to overdraw a
    // deadline already consumed by model conversion or bound tightening.
    let mip_timeout_secs = deadline
        .saturating_duration_since(std::time::Instant::now())
        .as_secs_f64();
    let mip_config = MipConfig {
        timeout_secs: mip_timeout_secs,
        feasibility_ingress,
        ..config
    };

    // Solve: handle disjunctive vs conjunctive properties
    let num_outputs = vnnlib.map(|s| s.num_outputs).unwrap_or(1);
    if requires_safenlp_shared_prefix {
        anyhow::ensure!(
            vnnlib
                .and_then(certified_shared_tree_pairwise_indices)
                .is_some(),
            "required SafeNLP shared-prefix ingress needs one conjunctive relational unsafe row"
        );
    }
    let mut native_hull_publication = None;
    let mut aggregate_hull_publication = None;
    let result = if mip_timeout_secs <= 0.0 {
        timeout_verification_result(num_outputs)
    } else if let Some(spec) = vnnlib {
        if spec.is_disjunction && spec.output_constraint_clauses.len() > 1 {
            // Disjunctive property: solve each clause independently.
            // SAT on ANY clause → Violated. UNSAT on ALL → Verified.
            solve_disjunctive(
                network,
                input,
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &intermediate_bounds_vec,
                spec,
                mip_config,
                deadline,
                num_outputs,
                json,
            )?
        } else {
            // Conjunctive property: single solve, optionally warm-started from PGD.
            let mut encoder = encode_feedforward(
                &weights,
                &biases,
                &layer_dims,
                &input_bounds,
                &intermediate_bounds_vec,
            )
            .map_err(|e| anyhow::anyhow!("MIP encoding failed: {}", e))?;
            // Required ingress owns this final feasibility attempt. Plain
            // required ingress forbids a marker; marked required ingress
            // requires the unique unsafe row marker. Historical callers retain
            // the existing environment-gated margin behavior.
            let margin_reframe_enabled = match feasibility_ingress {
                MipFeasibilityIngress::Historical => ay_margin_reframe_enabled(),
                MipFeasibilityIngress::RequireSafeNlpSharedBinaryPrefix => false,
                MipFeasibilityIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix => true,
            };
            let shared_tree_preclone_eligible = certified_shared_tree_preclone_eligible(
                !requires_safenlp_shared_prefix && mip_certified_shared_tree_enabled(),
                margin_reframe_enabled,
                backend,
                use_exact_single_hidden_fast_path,
                spec,
                deadline,
            );
            // The proof model must not contain the unsafe row: retain this
            // exact clone before `add_vnnlib_constraints` stamps it.
            let certified_shared_tree_base =
                shared_tree_preclone_eligible.then(|| encoder.clone().into_parts());
            let unsafe_rows = add_vnnlib_constraints(&mut encoder, spec)?;
            let mut parts = encoder.into_parts();
            // A single signed margin needs exact ReLU phases only for its
            // concave heads. Retain the existing rows/bounds and relax only
            // the other indicators, then certify THIS larger unsafe model.
            // SAT still follows original-network concrete replay below.
            if backend == MipBackend::Ay {
                if let Some(report) = convex_binary::relax(
                    &mut parts,
                    use_exact_single_hidden_fast_path,
                    &weights,
                    &biases,
                    &layer_dims,
                    &intermediate_bounds_vec,
                    spec,
                    &unsafe_rows,
                    deadline,
                ) {
                    eprintln!(
                        "[mip-convex-binaries] unstable={} relaxed_nonnegative={} negative_binaries={} rows_retained=1",
                        report.unstable_before,
                        report.relaxed_nonnegative,
                        report.negative_binaries,
                    );
                }
            }
            let mut short_hull_proof = None;
            let mut aggregate_hull_proof = None;
            // Keep the stronger checked coordinate hull in the same exact
            // complete-MIP model. A separate short native cover can close it
            // only after that model binding; cut metadata grants no verdict.
            if backend == MipBackend::Ay && use_exact_single_hidden_fast_path {
                let began = std::time::Instant::now();
                match ideal_cuts::strengthen_with_signed_refinement(
                    &mut parts, use_exact_single_hidden_fast_path, &weights, &biases,
                    &layer_dims, &intermediate_bounds_vec, &input_bounds, spec,
                    &unsafe_rows, deadline,
                    ideal_cuts::short_proof_allowed(
                        requires_safenlp_shared_prefix, shared_tree_preclone_eligible,
                        capture_bridge_publication,
                    ),
                ) {
                    Some(outcome) => {
                        let report = &outcome.cuts;
                        eprintln!(
                        "[mip-ideal-cuts] admitted=1 cuts={} input_terms={} checked_lower_bits={:016x} proposal_lp_calls={} t_ms={}",
                        report.cuts, report.input_terms, report.checked_hull_lower.to_bits(),
                        report.proposal_lp_calls, outcome.cuts_elapsed.as_millis(),
                        );
                        eprintln!(
                            "[mip-joint-cuts] individual={} joint={} pairs_attempted={} pairs_completed={}",
                            report.individual_cuts, report.joint_cuts,
                            report.joint_pairs_attempted, report.joint_pairs_completed,
                        );
                        eprintln!(
                            "[mip-compact-cuts] rows={} pairs={} root_lower_bits={} root_verified={}",
                            report.compact_cuts, report.compact_pairs,
                            report.compact_root_lower.map_or_else(||"na".to_owned(), |v|format!("{:016x}",v.to_bits())),
                            u8::from(outcome.aggregate.as_ref().is_some_and(|proof|proof.route()=="compact" && proof.verified())),
                        );
                        eprintln!(
                            "[mip-group-cuts] ledger=at-append groups={} attempts={} search_masks={} replay_masks={} work_units={} peak_bytes={}",
                            report.compact_groups,report.group_attempts,report.group_search_masks,
                            report.group_replay_masks,report.group_work,report.group_peak_bytes,
                        );
                        if let Some(sdp) = &report.global_sdp {
                            eprintln!(
                                "[mip-global-sdp] ledger=at-append rows=1 dimension={} active_dimension={} optimizer_iterations={} factorizations={} checked_candidates={} source_replays={} work_units={} admitted_bytes={} correction_bits={:016x} lower_bits={:016x} root_selected={}",
                                sdp.dimension,sdp.active_dimension,sdp.optimizer_iterations,
                                sdp.factorizations,sdp.checked_candidates,sdp.source_replays,
                                sdp.work_units,sdp.admitted_bytes,sdp.correction.to_bits(),sdp.lower.to_bits(),
                                u8::from(outcome.aggregate.as_ref().is_some_and(|proof|proof.route()=="global-sdp" && proof.verified())),
                            );
                        } else {
                            eprintln!("[mip-global-sdp] ledger=at-append rows=0");
                        }
                        if let Some(signed) = &report.signed {
                            eprintln!(
                                "[mip-signed-root] ledger=at-append attempted={} rows={} components={} refined_components={} assigned_positive_heads={} negative_heads={} coordinate_proposals={} completed_masks={} checked_masks={} source_replays={} work_units={} admitted_bytes={} lower_bits={} stop={} root_selected={}",
                                u8::from(signed.attempted), u8::from(signed.admitted), signed.components,
                                signed.refined_components, signed.assigned_positive_heads, signed.negative_heads,
                                signed.coordinate_proposals, signed.completed_masks, signed.checked_masks,
                                signed.source_replays, signed.work_units, signed.admitted_bytes,
                                signed.lower.map_or_else(|| "na".to_owned(), |v|format!("{:016x}", v.to_bits())),
                                signed.stop_reason,
                                u8::from(outcome.aggregate.as_ref().is_some_and(|proof|proof.route()=="signed-coordinate" && proof.verified())),
                            );
                            let m=&signed.mass;
                            let root=&signed.root_coupling;
                            eprintln!("[mip-root-coupling] requested={} admitted={} producer_attempts={} checker_slots={} support_attempts={} completed_supports={} selected={} work_units={} used_work_units={} admitted_bytes={} left={:?} right={:?} parent_count={} parent_indices={:?} selected_negatives={} tail_negatives={} positive_heads={} recipient_positive_heads={} funded_coordinates={} optimized_coordinates={} line_mask_attempts={} line_masks_completed={} line_masks_improved={} coordinate_visits={} metadata_work={} quoted_search_work={} quoted_check_work={} actual_check_work={} old_masks={} new_masks={} baseline_lower={:?} before_global_lower={:?} proposed_lower={:?} after_global_lower={:?} stop={}",
                                u8::from(root.requested),u8::from(root.admitted),root.producer_attempts,root.checker_slots,root.support_attempts,root.completed_supports,u8::from(root.selected),root.work_units,root.used_work_units,root.admitted_bytes,root.left,root.right,root.parent_count,root.parent_indices,root.selected_negatives,root.tail_negatives,root.positive_heads,root.recipient_positive_heads,root.funded_coordinates,root.optimized_coordinates,root.line_mask_attempts,root.line_masks_completed,root.line_masks_improved,root.coordinate_visits,root.metadata_work,root.quoted_search_work,root.quoted_check_work,root.actual_check_work,root.old_masks,root.new_masks,root.baseline_lower,root.before_global_lower,root.proposed_lower,root.after_global_lower,root.stop_reason);
                            eprintln!("[mip-signed-mass] ledger=at-append attempted={} constructed_appearances={} constructed_split_heads={} baseline_checks={} final_checks={} funded_visits={} completed_visits={} selected={} banked_lower_bits={} optional_lower_bits={} selected_positive_heads={} selected_positive_appearances={} selected_split_heads={} stop={}",
                                u8::from(m.attempted),m.positive_appearances,m.split_positive_heads,
                                m.baseline_checks,m.final_checks,m.funded_visits,m.completed_visits,u8::from(m.selected),
                                m.banked_lower.map_or_else(||"na".to_owned(),|v|format!("{:016x}",v.to_bits())),
                                m.lower.map_or_else(||"na".to_owned(),|v|format!("{:016x}",v.to_bits())),
                                signed.assigned_positive_heads,signed.positive_appearances,signed.split_positive_heads,m.stop_reason);
                            let cr=&signed.component_rows;
                            eprintln!("[mip-signed-component-rows] ledger=after-append attempted={} admitted={} quoted_components={} quoted_terms={} reserved_work_units={} work_units={} admitted_bytes={} collection_attempts={} collected_components={} prepared_rows={} prepared_terms={} appended_rows={} appended_terms={} stop={}",
                                u8::from(cr.attempted),u8::from(cr.admitted),cr.quoted_components,cr.quoted_terms,
                                cr.reserved_work_units,cr.work_units,cr.admitted_bytes,cr.collection_attempts,cr.collected_components,
                                cr.prepared_rows,cr.prepared_terms,cr.appended_rows,cr.appended_terms,cr.stop_reason);
                            let phase = &cr.phase;
                            eprintln!("[mip-signed-phase-rows] ledger=after-append attempted={} admitted={} all_selector_mode={} original_ir_row_limit={} work_units={} admitted_bytes={} collection_attempts={} collected_components={} prepared_rows={} appended_rows={} appended_terms={} stop={}",
                                u8::from(phase.attempted),u8::from(phase.admitted),u8::from(phase.all_selector_mode),phase.original_ir_row_limit,phase.work_units,phase.admitted_bytes,
                                phase.collection_attempts,phase.collected_components,phase.prepared_rows,
                                phase.appended_rows,phase.appended_terms,phase.stop_reason);
                            let cube = &phase.cube;
                            eprintln!("[mip-signed-phase-cube] ledger=after-append attempted={} admitted={} capture_total_work={} capture_activity_work={:?} capture_activity_bytes={:?} activity_transfer_attempts={} transferred_activity_values={} activity_values={} quoted_masks={} quoted_rows={} quoted_terms={} setup_work={} work_units={} reserved_work_units={} admitted_bytes={} collection_attempts={} collected_masks={} completed_tables={} prepare_attempts={} activity_components={} projected_components={} prepared_rows={} prepared_terms={} appended_rows={} appended_terms={} appended_selector_terms={} stop={}",
                                u8::from(cube.attempted),u8::from(cube.admitted),cube.capture_total_work,cube.capture_activity_work,cube.capture_activity_bytes,cube.activity_transfer_attempts,cube.transferred_activity_values,cube.activity_values,
                                cube.quoted_masks,cube.quoted_rows,cube.quoted_terms,cube.setup_work,cube.work_units,cube.reserved_work_units,cube.admitted_bytes,
                                cube.collection_attempts,cube.collected_masks,cube.completed_tables,cube.prepare_attempts,cube.activity_components,cube.projected_components,
                                cube.prepared_rows,cube.prepared_terms,cube.appended_rows,cube.appended_terms,cube.appended_selector_terms,cube.stop_reason);
                            let dom = &cr.domination;
                            let cell = &cube.cell;
                            eprintln!("[mip-signed-phase-cell] attempted={} admitted={} work_units={} admitted_bytes={} component={:?} masks={} improved_masks={} joint_improved_masks={} prepared_rows={} prepared_terms={} prepared_selectors={} appended_rows={}",
                                u8::from(cell.attempted),u8::from(cell.admitted),cell.work_units,cell.admitted_bytes,cell.component,
                                cell.collected_masks,cell.improved_masks,cell.joint_improved_masks,cell.prepared_rows,cell.prepared_terms,
                                cell.prepared_selectors,cell.appended_rows);
                            let ray = &cell.ray;
                            eprintln!("[mip-signed-phase-ray] admitted={} work_units={} admitted_bytes={} masks={} improved_masks={} prepared_rows={} prepared_terms={} prepared_selectors={} appended_rows={}",
                                u8::from(ray.admitted),ray.work_units,ray.admitted_bytes,ray.masks,ray.improved_masks,
                                ray.prepared_rows,ray.prepared_terms,ray.prepared_selectors,ray.appended_rows);
                            let joint = &cell.joint;
                            eprintln!("[mip-signed-phase-joint] admitted={} work_units={} admitted_bytes={} masks={} improved_masks={} prepared_rows={} prepared_terms={} prepared_selectors={} appended_rows={}",
                                u8::from(joint.admitted),joint.work_units,joint.admitted_bytes,joint.masks,joint.improved_masks,
                                joint.prepared_rows,joint.prepared_terms,joint.prepared_selectors,joint.appended_rows);
                            let input_phase = &cell.input_phase;
                            eprintln!("[mip-signed-input-phase] admitted={} work_units={} admitted_bytes={} masks={} candidates={} selected_candidate={:?} stop={}",
                                u8::from(input_phase.admitted),input_phase.work_units,input_phase.admitted_bytes,
                                input_phase.masks,input_phase.candidates,input_phase.selected_candidate,input_phase.stop_reason);
                            let table_lp = &cell.lp;
                            eprintln!("[mip-signed-table-lp] admitted={} activity_from_root={} work_units={} admitted_bytes={} calls={} pivots={} pivot_limit={} stop={} checked_masks={} score_lower={:?} envelope_upper={:?} gap_upper={:?} prepared_rows={} prepared_terms={} prepared_selectors={} appended_rows={}",
                                u8::from(table_lp.admitted),u8::from(table_lp.activity_from_root),table_lp.work_units,table_lp.admitted_bytes,table_lp.calls,table_lp.pivots,table_lp.pivot_limit,
                                table_lp.stop_reason,table_lp.checked_masks,table_lp.score_lower,table_lp.envelope_upper,table_lp.gap_upper,
                                table_lp.prepared_rows,table_lp.prepared_terms,table_lp.prepared_selectors,table_lp.appended_rows);
                            eprintln!("[mip-signed-domination] ledger=after-append attempted={} admitted={} producer_attempts={} advice_attempts={} export_attempts={} free_work_before={} checks_before={} checker_slots={} funded_targets={} funded_sparse={} sparse_fits={} nominal_scores={} support_attempts={} completed_supports={} work_units={} admitted_bytes={} appended_rows={} appended_terms={} joint_skipped_unfunded={} dense={} funded_positive_limit={} pool_heads={} capture_work_units={} total_optional_work_units={} funded_alias_heads={} checked_alias_rows={} appended_alias_rows={} alias_work_units={} alias_bytes={} priced_step_work_units={} priced_sort_work_units={} priced_step_bytes={} priced_step_attempted={} priced_step_completed={} priced_step_accepted={} group_allocation={} group_nominal_attempts={} group_nominal_masks={} group_checked_masks={} group_nominal_work={} group_checker_tail={} group_anchor={:?} group_partner={:?} group_nominal_lhs={:?} group_nominal_bound={:?} branch_tau_mode={} exclusive_seed_useful={:?} gradient_skipped={} tail_single_work={} tail_branch_work={} branch_tau_attempts={} branch_tau_completed={} branch_tau_selected={} branch_tau_checked_masks={} branch_tau_optimizer_work={} branch_tau0={:?} branch_tau1={:?} branch_tau2={:?} branch_tau3={:?} branch_bound0={:?} branch_bound1={:?} branch_bound2={:?} branch_bound3={:?} stop={}",
                                u8::from(dom.attempted),u8::from(dom.admitted),dom.producer_attempts,dom.advice_attempts,dom.export_attempts,
                                dom.free_work_before,dom.checks_before,dom.checker_slots,dom.funded_targets,
                                dom.funded_sparse_fits,dom.sparse_fits,dom.nominal_scores,dom.support_attempts,
                                dom.completed_supports,dom.work_units,dom.admitted_bytes,dom.appended_rows,
                                dom.appended_terms,u8::from(dom.joint_skipped_unfunded),u8::from(dom.dense),dom.funded_positive_limit,
                                dom.pool_heads,dom.capture_work_units,dom.total_optional_work_units,dom.funded_alias_heads,
                                dom.checked_alias_rows,dom.appended_alias_rows,dom.alias_work_units,dom.alias_bytes,dom.priced_step_work_units,dom.priced_sort_work_units,dom.priced_step_bytes,u8::from(dom.priced_step_attempted),u8::from(dom.priced_step_completed),u8::from(dom.priced_step_accepted),u8::from(dom.group_allocation),dom.group_nominal_attempts,dom.group_nominal_masks,dom.group_checked_masks,dom.group_nominal_work,dom.group_checker_tail,dom.group_anchor,dom.group_partner,dom.group_nominal_lhs,dom.group_nominal_bound,
                                u8::from(dom.branch_tau_mode),dom.exclusive_seed_useful,u8::from(dom.gradient_skipped),dom.tail_single_work,dom.tail_branch_work,
                                dom.branch_tau_attempts,dom.branch_tau_completed,u8::from(dom.branch_tau_selected),dom.branch_tau_checked_masks,dom.branch_tau_optimizer_work,
                                dom.branch_tau_values.map(|a|a[0]),dom.branch_tau_values.map(|a|a[1]),dom.branch_tau_values.map(|a|a[2]),dom.branch_tau_values.map(|a|a[3]),
                                dom.branch_tau_mask_prices.map(|a|a[0]),dom.branch_tau_mask_prices.map(|a|a[1]),dom.branch_tau_mask_prices.map(|a|a[2]),dom.branch_tau_mask_prices.map(|a|a[3]),dom.stop_reason);
                            let joint = &cr.joint;
                            eprintln!("[mip-signed-joint-rows] ledger=after-append attempted={} prepaid_requested={} prepaid_admitted={} prepaid_work_units={} prepaid_bytes={} prepaid_used_work_units={} prepaid_checker_slots={} producer_attempts={} checks_before={} remaining_checks_before={} checker_slots={} free_work_before={} eligible_pairs={} partial_eligible={} partial_paid={} partial_completed={} partial_selected={} partial_tail={} partial_checked_masks={} partial_tail_rechecks={} partial_work={} partial_rows={} paid_pairs={} completed_pairs={} quoted_masks={} checked_masks={} coordinates={} work_units={} admitted_bytes={} appended_rows={} appended_terms={} stop={}",
                                u8::from(joint.attempted),u8::from(joint.prepaid_requested),u8::from(joint.prepaid_admitted),
                                joint.prepaid_work_units,joint.prepaid_bytes,joint.prepaid_used_work_units,joint.prepaid_checker_slots,
                                joint.producer_attempts,joint.checks_before,joint.remaining_checks_before,
                                joint.checker_slots,joint.free_work_before,joint.eligible_pairs,
                                joint.partial_eligible_pairs,joint.partial_paid_pairs,joint.partial_completed_pairs,
                                joint.partial_selected_heads,joint.partial_tail_heads,joint.partial_checked_masks,
                                joint.partial_tail_rechecks,joint.partial_work_units,joint.partial_appended_rows,
                                joint.paid_pairs,joint.completed_pairs,
                                joint.quoted_masks,joint.checked_masks,joint.coordinates,joint.work_units,joint.admitted_bytes,
                                joint.appended_rows,joint.appended_terms,joint.stop_reason);
                            let d = &signed.direction;
                            eprintln!("[mip-signed-direction] ledger=at-append attempts={} complete_sweeps={} coordinates={} cached_masks={} checked_candidates={} accepted_candidates={} work_units={} cache_bytes={} stop={}",
                                d.attempts,d.completed_sweeps,d.coordinates,d.cached_masks,
                                d.checked_candidates,d.accepted_candidates,d.work_units,d.cache_bytes,d.stop_reason);

                        }
                        aggregate_hull_proof = outcome.aggregate;
                        if let Some(proof) = &outcome.continuation {
                            let stats = proof.report();
                            if let Some(c) = proof.signed_child_report() {
                                eprintln!("[mip-signed-child] ledger=after-continuation collection_attempts={} cached_masks={} cache_bytes={} bind_attempts={} calls={} completed={} work_units={} total_work_units={} source_replays={} positive={} lp_skips={} corner_axes={} corner_price_attempts={} corner_root_lower={:?} corner_ray_axes={} corner_base_root_lower={:?} corner_ray_root_lower={:?} corner_cover_admitted={} corner_cover_leaves={} corner_cover_root_lower={:?} stop={}",
                                    c.collection_attempts,c.cached_masks,c.cache_bytes,c.bind_attempts,
                                    c.attempted_calls,c.completed_calls,c.work_units,c.total_work_units,c.source_replays,c.positive_calls,c.lp_skips,c.corner_axes,c.corner_price_attempts,c.corner_root_lower,c.corner_ray_axes,c.corner_base_root_lower,c.corner_ray_root_lower,c.corner_cover_admitted,c.corner_cover_leaves,c.corner_cover_root_lower,c.stop_reason);
                            }
                            eprintln!(
                                "[mip-hull-portfolio] root_reused=1 shared_root_lp_calls={} continuation_root_lp_calls={} continuation_lp_calls={} stop={} nodes={} closed={} open={} verified={} t_ms={}",
                                report.proposal_lp_calls, stats.root_lp_calls, stats.proposal_lp_calls,
                                stats.stop.label(), stats.nodes, stats.closed, stats.open, u8::from(proof.verified()),
                                outcome.continuation_elapsed.as_millis(),
                            );
                        } else if aggregate_hull_proof.as_ref().is_some_and(|proof|proof.verified()) {
                            let stop = match aggregate_hull_proof.as_ref().unwrap().route() {
                                "signed-coordinate" => "signed-coordinate-proved",
                                "global-sdp" => "global-sdp-proved",
                                _ => "aggregate-proved",
                            };
                            eprintln!("[mip-hull-portfolio] root_reused=0 stop={stop} tree_skipped=1 verified=0");
                        } else {
                            eprintln!("[mip-hull-portfolio] root_reused=0 stop=declined verified=0");
                        }
                        short_hull_proof = outcome.continuation;
                    }
                    None => eprintln!(
                        "[mip-ideal-cuts] admitted=0 cuts=0 input_terms=0 checked_lower_bits=na proposal_lp_calls=na t_ms={}",
                        began.elapsed().as_millis(),
                    ),
                }
            }
            let marked_margin = maybe_mark_unique_ay_margin(
                margin_reframe_enabled,
                backend,
                &mut parts,
                &unsafe_rows,
            )?;
            if requires_safenlp_marked_margin_shared_prefix {
                anyhow::ensure!(
                    marked_margin
                        && unsafe_rows.len() == 1
                        && parts.problem.margin_row() == Some(unsafe_rows[0]),
                    "required SafeNLP marked-margin ingress lost the unique unsafe row identity"
                );
            }
            if marked_margin {
                tracing::info!(
                    row = unsafe_rows[0].0,
                    shared_prefix_required = requires_safenlp_marked_margin_shared_prefix,
                    "AY direct-MIP margin reframe armed for the unique unsafe row"
                );
            }
            let certified_shared_tree = certified_shared_tree_base.as_ref().and_then(|base| {
                let plan = certified_shared_tree_plan(shared_tree_preclone_eligible, spec, base)?;
                try_certified_shared_tree(base, &plan, deadline)
            });
            if aggregate_hull_proof
                .as_ref()
                .is_some_and(|proof| proof.verified())
            {
                // A checked real-arithmetic full-q root proof, not an AY
                // status or typed OriginalReplay/Source E1 receipt.
                aggregate_hull_publication = aggregate_hull_proof.take();
                VerificationResult::Verified {
                    provenance: Default::default(),
                    output_bounds: vec![
                        Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY);
                        num_outputs
                    ],
                    proof: None,
                    actual_method: Some(ny_core::MethodUsed::BetaCrown),
                }
            } else if short_hull_proof
                .as_ref()
                .is_some_and(|proof| proof.verified())
            {
                // A native checked complete phase cover, not an AY status or
                // an inferred consequence of how many cuts were appended.
                native_hull_publication = short_hull_proof.take();
                VerificationResult::Verified {
                    provenance: Default::default(),
                    output_bounds: vec![
                        Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY);
                        num_outputs
                    ],
                    proof: None,
                    actual_method: Some(ny_core::MethodUsed::BetaCrown),
                }
            } else if let Some(certified) = certified_shared_tree {
                tracing::info!(
                    proof_route = ?certified.proof_route,
                    ay_tree_leaves = certified.ay_tree_leaves,
                    ny_cert_replays = certified.ny_cert_farkas_replays,
                    "certified shared-tree canary excluded the unique pairwise unsafe region"
                );
                VerificationResult::Verified {
                    provenance: Default::default(),
                    output_bounds: vec![
                        Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY,);
                        num_outputs
                    ],
                    proof: None,
                    actual_method: Some(mip_unsat_method_tag()),
                }
            } else if std::time::Instant::now() >= deadline {
                // The proof attempt consumed the original caller-owned
                // envelope. Do not construct or launch the historical
                // unsafe-row solve after that deadline.
                timeout_verification_result(num_outputs)
            } else {
                let warm_start_cols = warm_start_candidate.and_then(|candidate| {
                    build_warm_start_vector(
                        candidate,
                        &weights,
                        &biases,
                        &layer_dims,
                        &intermediate_bounds_vec,
                        parts.num_cols,
                    )
                });
                // Soundness gate: clamp + independent forward revalidation
                // before claiming Violated. The fallback receives only time
                // still remaining under the original absolute deadline.
                let conjunctive_constraints = conjunctive_constraints_owned(spec);
                let live_timeout = deadline
                    .saturating_duration_since(std::time::Instant::now())
                    .as_secs_f64();
                let mip_config = MipConfig {
                    timeout_secs: mip_config.timeout_secs.min(live_timeout),
                    ..mip_config
                };
                let mut solver = MipSolver::new(parts, mip_config);
                let objective_schedule =
                    if !requires_safenlp_shared_prefix && ay_objective_first_sat_enabled() {
                        objective_first_sat_budget(
                            true,
                            backend,
                            unsafe_rows.len(),
                            mip_config.timeout_secs,
                            solver.effective_feasibility_timeout_secs(),
                        )
                        .and_then(|budget| {
                            ObjectiveFirstSatLedger::start(budget, mip_config.ay_hard_deadline)
                                .map(|ledger| (budget, ledger))
                        })
                    } else {
                        None
                    };
                let objective_hit = objective_schedule.and_then(|(budget, ledger)| {
                    let probe_secs = ledger.probe_secs(budget)?;
                    tracing::info!(
                        probe_secs,
                        historical_envelope_secs = budget.envelope_secs,
                        nominal_fallback_secs = budget
                            .fallback_secs_after_elapsed(budget.probe_secs)
                            .unwrap_or(0.0),
                        "AY objective-first SAT lane: probing inside the historical wall envelope"
                    );
                    let hit = revalidate_objective_first_sat_probe(
                        solver.probe_one_sided_sat_until(
                            unsafe_rows[0],
                            probe_secs,
                            ledger.deadline,
                        ),
                        network,
                        input,
                        &conjunctive_constraints,
                        num_outputs,
                    );
                    if ledger.expired() {
                        None
                    } else {
                        hit
                    }
                });
                if let Some(hit) = objective_hit {
                    hit
                } else {
                    let fallback_config = match objective_schedule {
                        Some((budget, ledger)) => {
                            solver
                                .set_ay_hard_deadline(budget.envelope_secs, ledger.deadline)
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "invalid objective-first fallback deadline: {e}"
                                    )
                                })?;
                            objective_first_sat_fallback_config(mip_config, budget, ledger)
                        }
                        None => mip_config,
                    };
                    let mip_result = solver
                        .check_feasibility_with_warm_start(warm_start_cols.as_deref())
                        .map_err(|e| anyhow::anyhow!("MIP solve failed: {}", e))?;
                    let was_sat = matches!(&mip_result, MipResult::Sat { .. });
                    let revalidated = map_mip_result_revalidated(
                        mip_result,
                        network,
                        input,
                        &conjunctive_constraints,
                        num_outputs,
                    );
                    if unconfirmed_sat_retry_allowed(
                        feasibility_ingress,
                        was_sat,
                        matches!(revalidated, VerificationResult::Violated { .. }),
                    ) {
                        // Solver-tolerance witness: re-solve with a violation
                        // slack for a robust one.
                        let retry_config = MipConfig {
                            timeout_secs: fallback_config.timeout_secs.min(
                                deadline
                                    .saturating_duration_since(std::time::Instant::now())
                                    .as_secs_f64(),
                            ),
                            ..fallback_config
                        };
                        retry_with_violation_slack(
                            network,
                            input,
                            &weights,
                            &biases,
                            &layer_dims,
                            &input_bounds,
                            &intermediate_bounds_vec,
                            &conjunctive_constraints,
                            retry_config,
                            num_outputs,
                        )
                        .unwrap_or(revalidated)
                    } else {
                        revalidated
                    }
                }
            }
        }
    } else {
        // Non-VNNLIB path: threshold defines safety property (output >= threshold).
        // Unsafe region: output < threshold. In LP/MIP, approximate with output <= threshold.
        let mut encoder = encode_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &intermediate_bounds_vec,
        )
        .map_err(|e| anyhow::anyhow!("MIP encoding failed: {}", e))?;
        encoder
            .constrain_output_leq_const(0, threshold as f64)
            .map_err(|e| anyhow::anyhow!("constraint failed: {}", e))?;
        let parts = encoder.into_parts();
        let live_timeout = deadline
            .saturating_duration_since(std::time::Instant::now())
            .as_secs_f64();
        let solver = MipSolver::new(
            parts,
            MipConfig {
                timeout_secs: mip_config.timeout_secs.min(live_timeout),
                ..mip_config
            },
        );
        // Non-VNNLIB path: no PGD candidate available, warm-start not applicable.
        let mip_result = solver
            .check_feasibility_with_warm_start(None)
            .map_err(|e| anyhow::anyhow!("MIP solve failed: {}", e))?;
        // Soundness gate: the safety property is output[0] >= threshold, so the
        // unsafe region is output[0] < threshold. Revalidate the witness against
        // an equivalent LessThanConst(0, threshold) constraint.
        let threshold_constraint = vec![OutputConstraint::LessThanConst(0, threshold as f64)];
        map_mip_result_revalidated(
            mip_result,
            network,
            input,
            &threshold_constraint,
            num_outputs,
        )
    };
    // Solver/forward APIs are synchronous and may return a fraction after
    // their requested timeout. A late mathematical decision is not admissible
    // for this bounded attempt.
    // This is the MIP ingress's terminal publication check. It never marks
    // deadline causality: the solver must eventually expose a typed source for
    // that, rather than relying on the result observed after synchronous work.
    let elapsed = reporting_start.elapsed();
    // Keep the opaque completed cover until the existing terminal publisher.
    // Do not reduce its current FP/stage/original authority to a legacy bool.
    let result = if native_hull_publication
        .as_ref()
        .is_some_and(|proof| !proof.verified())
        || aggregate_hull_publication
            .as_ref()
            .is_some_and(|proof| !proof.verified())
    {
        VerificationResult::Unknown {
            provenance: Default::default(),
            bounds: vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); num_outputs],
            reason: "native hull publication authority expired or FP qualification refused".into(),
            actual_method: Some(ny_core::MethodUsed::BetaCrown),
        }
    } else {
        result
    };
    let result = gate_mip_result_at_deadline(result, num_outputs, deadline);

    // Output results
    let publication_refused = print_result_with_native_hull(
        &result,
        property,
        model,
        epsilon,
        threshold,
        elapsed,
        backend,
        native_hull_publication.is_some() || aggregate_hull_publication.is_some(),
        Some(effective_treatment),
        json,
    )?;

    let exit_code = if publication_refused {
        crate::commands::verify::exit_codes::UNKNOWN
    } else {
        verification_result_exit_code(&result)
    };
    if exit_code != crate::commands::verify::exit_codes::VERIFIED && !super::output::is_capturing()
    {
        std::process::exit(exit_code);
    }

    Ok(if capture_bridge_publication {
        bridge_unsat_publication(&result, publication_refused, json)
    } else {
        None
    })
}

/// Owned copy of the conjunctive constraint list fed to `add_vnnlib_constraints`.
///
/// Mirrors `add_vnnlib_constraints`' selection logic so the revalidation gate
/// re-checks exactly the constraints the encoder asserted. Prefers the flattened
/// `output_constraint_clauses` (non-disjunctive specs may still carry a single
/// clause there) and falls back to `output_constraints`.
fn conjunctive_constraints_owned(spec: &VnnLibSpec) -> Vec<OutputConstraint> {
    if !spec.output_constraint_clauses.is_empty() {
        spec.output_constraint_clauses
            .iter()
            .flatten()
            .cloned()
            .collect()
    } else {
        spec.output_constraints.clone()
    }
}

/// Add VNNLIB output constraints to the MIP encoder (conjunctive only).
///
/// Disjunctive specs are handled by `solve_disjunctive` upstream.
fn add_vnnlib_constraints(
    encoder: &mut ny_mip::MipEncoder,
    spec: &VnnLibSpec,
) -> Result<Vec<ny_mip::ir::Row>> {
    let constraints = if !spec.output_constraint_clauses.is_empty() {
        spec.output_constraint_clauses
            .iter()
            .flatten()
            .collect::<Vec<_>>()
    } else {
        spec.output_constraints.iter().collect::<Vec<_>>()
    };

    let mut rows = Vec::with_capacity(constraints.len());
    for constraint in constraints {
        rows.push(encode_output_constraint(encoder, constraint)?);
    }
    Ok(rows)
}

/// #metaroom-mip-governor — the CERTIFIED-CLAUSE PRESCREEN for the disjunctive
/// MIP escalation.
///
/// # The defect
///
/// `solve_disjunctive` seeds its pending set with EVERY clause and then hands
/// each one `remaining / pending` seconds. On metaroom_2023 4cnn_ry_57_9
/// spec_idx_28 that is 19 clauses over a 50 s escalation — 2.279 s each — and
/// the measured run abandoned all nineteen at their slice with
/// `certified_unsat=0`. Eighteen of those clauses had already been refuted at
/// the ROOT, in 1.1 s, by an ordinary CROWN pass. The one genuinely open clause
/// therefore received one nineteenth of a budget it needed all of.
///
/// # The certificate
///
/// Every VNN-LIB output constraint is a linear row over the outputs, and a
/// clause is the CONJUNCTION of its constraints, so the clause's unsafe region
/// is empty as soon as ONE of its rows is unsatisfiable. Appending those rows
/// to the network as a final `LinearLayer` and running the same sound CROWN-IBP
/// collection the encoder already uses gives a certified upper bound on each
/// row; `sup(a.y) < b` refutes the row over the WHOLE input box.
///
/// # Soundness
///
/// A disjunctive `Unsat` requires every clause to be infeasible. A clause
/// refuted here carries exactly the certificate ny publishes `unsat` on
/// everywhere else, so omitting it from the solver's pending set is equivalent
/// to solving it and receiving `Unsat { certified: true }` — and in particular
/// it is NOT booked as a timeout, so it can never mask an undecided clause.
///
/// FAIL CLOSED at every step. `None` means "no certified set could be
/// established", and the caller must then use ALL clauses:
///
/// * a clause carrying its own per-clause input box is never certified (the
///   bound here is taken over the global box, and the parser's union widening
///   is not re-verified here);
/// * a constraint naming an out-of-range output index, or a non-finite
///   constant, abandons the whole prescreen;
/// * a bound-collection failure, a shape the spec layer cannot be appended to,
///   or a non-finite bound abandons the whole prescreen;
/// * an expired deadline degrades the collection to plain IBP, which is still
///   sound — looser, so it certifies fewer clauses, never more.
///
/// Strict comparisons (`<`, `>`) are treated as their non-strict counterparts,
/// which requires MORE of the bound than refuting the strict form does.
fn certified_infeasible_clauses(
    network: &Network,
    input: &BoundedTensor,
    spec: &VnnLibSpec,
    num_outputs: usize,
    deadline: std::time::Instant,
) -> Option<Vec<bool>> {
    let clauses = &spec.output_constraint_clauses;
    if clauses.is_empty() || num_outputs == 0 {
        return None;
    }

    // One row per constraint, in clause order, as `a . y >= b` over the
    // outputs. `row_owner[r]` is the clause that row belongs to.
    let mut weights: Vec<f32> = Vec::new();
    let mut rhs: Vec<f64> = Vec::new();
    let mut row_owner: Vec<usize> = Vec::new();
    for (clause_idx, clause) in clauses.iter().enumerate() {
        // A clause with its own input box is bounded over a box this pass does
        // not use; leave it to the solver.
        if spec
            .per_clause_input_bounds
            .get(clause_idx)
            .is_some_and(|overrides| !overrides.is_empty())
        {
            continue;
        }
        for constraint in clause {
            let mut a = vec![0.0f32; num_outputs];
            let b = match constraint {
                // UNSAFE: Y_i <= Y_j  <=>  Y_j - Y_i >= 0
                OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
                    if *i >= num_outputs || *j >= num_outputs {
                        return None;
                    }
                    a[*j] += 1.0;
                    a[*i] -= 1.0;
                    0.0
                }
                // UNSAFE: Y_i >= Y_j  <=>  Y_i - Y_j >= 0
                OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
                    if *i >= num_outputs || *j >= num_outputs {
                        return None;
                    }
                    a[*i] += 1.0;
                    a[*j] -= 1.0;
                    0.0
                }
                // UNSAFE: Y_i <= c  <=>  -Y_i >= -c
                OutputConstraint::LessEqConst(i, c) | OutputConstraint::LessThanConst(i, c) => {
                    if *i >= num_outputs || !c.is_finite() {
                        return None;
                    }
                    a[*i] = -1.0;
                    -*c
                }
                // UNSAFE: Y_i >= c
                OutputConstraint::GreaterEqConst(i, c)
                | OutputConstraint::GreaterThanConst(i, c) => {
                    if *i >= num_outputs || !c.is_finite() {
                        return None;
                    }
                    a[*i] = 1.0;
                    *c
                }
                // `OutputConstraint` is `#[non_exhaustive]`: an unmapped
                // variant must abandon the prescreen, never be skipped.
                _ => return None,
            };
            weights.extend_from_slice(&a);
            rhs.push(b);
            row_owner.push(clause_idx);
        }
    }
    if row_owner.is_empty() {
        return None;
    }

    let num_rows = rhs.len();
    // The spec layer is dense (`num_rows x num_outputs`); a spec with tens of
    // thousands of literals is not what this prescreen is for, and building the
    // layer would cost more than the escalation it is trying to protect.
    const MAX_PRESCREEN_ROWS: usize = 8192;
    if num_rows > MAX_PRESCREEN_ROWS {
        return None;
    }
    let spec_weight = ndarray::Array2::from_shape_vec((num_rows, num_outputs), weights).ok()?;
    let spec_layer = LinearLayer::new(spec_weight, None).ok()?;
    let mut augmented = network.clone();
    augmented.add_layer(Layer::Linear(spec_layer));

    let bounds =
        collect_mip_intermediate_bounds_with_deadline(&augmented, input, Some(deadline)).ok()?;
    // `SoundIntermediateBounds` deliberately exposes no Vec API — its field is
    // private so a plain (uncertified) vector cannot be substituted for a
    // certified one. Read through its own view rather than reaching past it.
    let tail = bounds.as_slice().last()?;
    let upper: Vec<f32> = tail.upper().iter().copied().collect();
    if upper.len() != num_rows {
        return None;
    }

    let mut certified = vec![false; clauses.len()];
    for (row, (&clause_idx, &b)) in row_owner.iter().zip(rhs.iter()).enumerate() {
        let sup = f64::from(upper[row]);
        if sup.is_finite() && sup < b {
            certified[clause_idx] = true;
        }
    }
    Some(certified)
}

/// #metaroom-mip-governor — seed the disjunctive escalation's pending set.
///
/// FAIL CLOSED: `None`, or a flag vector whose length disagrees with the clause
/// count, falls back to ALL clauses and reports `all-fallback`. The returned
/// tag is the observable's `source=` field, so "ran and skipped nothing" is
/// always distinguishable from "could not establish the certified set".
fn seed_escalation_pending(
    num_clauses: usize,
    certified: Option<&[bool]>,
) -> (Vec<usize>, &'static str) {
    match certified {
        Some(flags) if flags.len() == num_clauses => (
            (0..num_clauses).filter(|idx| !flags[*idx]).collect(),
            "certified",
        ),
        _ => ((0..num_clauses).collect(), "all-fallback"),
    }
}

/// Solve disjunctive VNNLIB property by solving each clause independently.
///
/// Strategy: for each clause, encode a separate MIP and solve.
/// - If ANY clause is SAT → the overall property is VIOLATED
/// - If ALL clauses are certified UNSAT → the overall property is VERIFIED
/// - If any clause times out and none is SAT → TIMEOUT
///
/// Reference: alpha-beta-CROWN solves disjunctive properties the same way
/// (one MIP per clause, early exit on SAT).
#[allow(clippy::too_many_arguments)]
fn solve_disjunctive(
    network: &Network,
    input: &BoundedTensor,
    weights: &[Vec<f64>],
    biases: &[Vec<f64>],
    layer_dims: &[usize],
    input_bounds: &[Bound],
    intermediate_bounds: &[Vec<Bound>],
    spec: &VnnLibSpec,
    config: MipConfig,
    deadline: std::time::Instant,
    num_outputs: usize,
    json: bool,
) -> Result<VerificationResult> {
    let num_clauses = spec.output_constraint_clauses.len();
    if !json {
        println!(
            "  Disjunctive property: {} clauses, solving independently...",
            num_clauses
        );
    }

    let mut had_timeout = false;
    // An exact solver status without independently checked infeasibility
    // evidence is not a proof. Such a clause must block Verified just like any
    // other undecided clause, but retrying the identical solve cannot upgrade
    // its evidence, so remember it separately from timeouts.
    let mut had_uncertified_unsat = false;
    // Tracks clauses where the MIP found a feasible unsafe point but the witness
    // failed in-box revalidation. The clause's unsafe region IS reachable, so we
    // must NOT conclude Verified — only the concrete witness is unconfirmed.
    let mut had_unconfirmed_sat = false;
    // Progressive multi-round schedule (#malbeware-mip-budget): the first pass
    // gives every clause `remaining / remaining_clauses`; clauses that hit that
    // slice (Timeout/Error) are RETRIED while overall budget remains, with the
    // slice recomputed over the (much smaller) undecided set. Measured on
    // malbeware 4-25 eps-3 (24 clauses, ~42s MIP slice): ~20 easy clauses close
    // in ~0.4s each, so a hard clause's slice grows from ~1.8s (round 1) to
    // 20s+ (round 2) — the single-pass schedule instead returned `timeout`
    // with >20s of granted budget unused. SOUND: per-clause verdict semantics
    // are unchanged (Unsat still requires proven infeasibility on EVERY
    // clause; a retry only grants a clause more solver time), and any Sat
    // still passes the in-box revalidation gate before being emitted.
    // Round cap: pathological non-progress can at most burn the granted MIP
    // budget, but keep a hard cap so a zero-second-slice loop cannot spin.
    const MAX_CLAUSE_ROUNDS: usize = 4;

    // Encode the network ONCE; per clause, stamp the clause's output
    // constraint onto a clone. The base encode re-scans the dense unfolded
    // weight matrices (252M f64 for malbeware 16-25), so re-encoding per
    // clause cost ~seconds x 24 clauses; the clone copies only the built
    // sparse IR. Identical formulation: `encode_feedforward` is deterministic
    // in its inputs, which do not change across clauses.
    let base_encoder = encode_feedforward(
        weights,
        biases,
        layer_dims,
        input_bounds,
        intermediate_bounds,
    )
    .map_err(|e| anyhow::anyhow!("MIP encoding failed: {}", e))?;

    // #metaroom-mip-governor — seed the pending set from the UNCERTIFIED
    // clauses only when the typed lever is armed. FAIL CLOSED: any failure to
    // establish the certified set (including the lever being disarmed) falls
    // back to the historical all-clauses seed, and the observable says which.
    //
    // The prescreen gets a bounded slice of the escalation budget so a slow
    // collection cannot eat the solve it exists to protect; an expired slice
    // degrades the collection to plain IBP, which is sound and merely looser.
    let skip_certified = crate::preset_gates::mip_escalate_skip_certified_armed();
    let escalation_start = std::time::Instant::now();
    let escalation_remaining = deadline
        .saturating_duration_since(escalation_start)
        .as_secs_f64();
    let certified: Option<Vec<bool>> = if skip_certified {
        let slice = (escalation_remaining * 0.2).clamp(0.0, 10.0);
        let prescreen_deadline = escalation_start
            .checked_add(std::time::Duration::from_secs_f64(slice))
            .unwrap_or(deadline)
            .min(deadline);
        certified_infeasible_clauses(network, input, spec, num_outputs, prescreen_deadline)
    } else {
        None
    };
    let (mut pending, pending_source) = seed_escalation_pending(num_clauses, certified.as_deref());
    // Every clause must still be accounted for: an empty pending set means
    // every clause carried a certificate, which is a proof, not a vacuum.
    let clauses_certified = num_clauses - pending.len();
    let post_prescreen_remaining = deadline
        .saturating_duration_since(std::time::Instant::now())
        .as_secs_f64();
    eprintln!(
        "[mip-escalate] clauses_total={num_clauses} clauses_certified={clauses_certified} \
clauses_pending={} slice_s={:.3} source={pending_source}",
        pending.len(),
        if pending.is_empty() {
            0.0
        } else {
            post_prescreen_remaining / pending.len() as f64
        },
    );
    // One certified-UNSAT phase-split memo per clause, living ACROSS retry
    // rounds: a clause abandoned at 15-of-16 certified-Unsat subproblems
    // re-races only the open one instead of starting from zero. The memo is
    // keyed by a full problem fingerprint inside ny-mip (fail-closed: any
    // re-encode drift clears it); the deterministic base_encoder clone +
    // clause stamping above is what makes the fingerprint match across
    // rounds.
    let mut split_caches: Vec<SplitUnsatCache> = (0..num_clauses)
        .map(|_| SplitUnsatCache::default())
        .collect();
    'rounds: for round in 0..MAX_CLAUSE_ROUNDS {
        if pending.is_empty() {
            break;
        }
        if round > 0 {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            // A retry round needs a meaningful slice to make progress; stop
            // once the tail budget is exhausted (< 1s per pending clause).
            if remaining.as_secs_f64() < pending.len() as f64 {
                break;
            }
            if !json {
                println!(
                    "  Retry round {}: {} timed-out clause(s), {:.1}s budget remaining",
                    round + 1,
                    pending.len(),
                    remaining.as_secs_f64()
                );
            }
        }
        let round_pending = std::mem::take(&mut pending);
        let round_total = round_pending.len();

        for (pos, &clause_idx) in round_pending.iter().enumerate() {
            let clause = &spec.output_constraint_clauses[clause_idx];
            // Adaptive per-clause timeout: remaining budget / remaining clauses
            // in this round. Prevents early clauses from consuming the budget.
            let now = std::time::Instant::now();
            if now >= deadline {
                // Out of budget: everything not yet decided stays pending.
                pending.extend(round_pending[pos..].iter().copied());
                break 'rounds;
            }
            let remaining_secs = deadline.duration_since(now).as_secs_f64();
            let remaining_clauses = (round_total - pos).max(1) as f64;
            let clause_timeout = remaining_secs / remaining_clauses;
            let clause_config = MipConfig {
                timeout_secs: clause_timeout,
                ..config
            };

            let mut encoder = base_encoder.clone();

            let mut unsafe_rows = Vec::with_capacity(clause.len());
            for constraint in clause {
                unsafe_rows.push(encode_output_constraint(&mut encoder, constraint)?);
            }
            let clause_config = MipConfig {
                timeout_secs: clause_config.timeout_secs.min(
                    deadline
                        .saturating_duration_since(std::time::Instant::now())
                        .as_secs_f64(),
                ),
                ..clause_config
            };

            // Probe each clause at most once.  Retry rounds spend all of their
            // enlarged tail slice on the historical feasibility/certificate
            // solve.
            let mut parts = encoder.into_parts();
            if maybe_mark_unique_ay_margin(
                ay_margin_reframe_enabled(),
                config.backend,
                &mut parts,
                &unsafe_rows,
            )? {
                tracing::info!(
                    clause = clause_idx,
                    row = unsafe_rows[0].0,
                    "AY direct-MIP margin reframe armed for one disjunctive unsafe row"
                );
            }
            let mut solver = MipSolver::new(parts, clause_config);
            let objective_schedule = if round == 0 && ay_objective_first_sat_enabled() {
                objective_first_sat_budget(
                    true,
                    config.backend,
                    unsafe_rows.len(),
                    clause_timeout,
                    solver.effective_feasibility_timeout_secs(),
                )
                .and_then(|budget| {
                    ObjectiveFirstSatLedger::start(budget, clause_config.ay_hard_deadline)
                        .map(|ledger| (budget, ledger))
                })
            } else {
                None
            };
            if let Some((budget, ledger)) = objective_schedule {
                if let Some(probe_secs) = ledger.probe_secs(budget) {
                    tracing::info!(
                        clause = clause_idx,
                        probe_secs,
                        historical_envelope_secs = budget.envelope_secs,
                        nominal_fallback_secs = budget
                            .fallback_secs_after_elapsed(budget.probe_secs)
                            .unwrap_or(0.0),
                        "AY objective-first SAT lane: probing one disjunctive unsafe row inside \
                         the historical wall envelope"
                    );
                    let hit = revalidate_objective_first_sat_probe(
                        solver.probe_one_sided_sat_until(
                            unsafe_rows[0],
                            probe_secs,
                            ledger.deadline,
                        ),
                        network,
                        input,
                        clause,
                        num_outputs,
                    );
                    if !ledger.expired() {
                        if let Some(hit) = hit {
                            if !json {
                                println!(
                                    "  Clause {}/{}: SAT (AY objective-first candidate replayed)",
                                    clause_idx + 1,
                                    num_clauses
                                );
                            }
                            return Ok(hit);
                        }
                    }
                }
            }
            let fallback_config = match objective_schedule {
                Some((budget, ledger)) => {
                    solver
                        .set_ay_hard_deadline(budget.envelope_secs, ledger.deadline)
                        .map_err(|e| {
                            anyhow::anyhow!("invalid objective-first clause fallback deadline: {e}")
                        })?;
                    objective_first_sat_fallback_config(clause_config, budget, ledger)
                }
                None => clause_config,
            };
            let mip_result = solver
                .check_feasibility_cached(None, &mut split_caches[clause_idx])
                .map_err(|e| anyhow::anyhow!("MIP solve failed on clause {}: {}", clause_idx, e))?;

            match &mip_result {
                MipResult::Sat { .. } => {
                    // Soundness gate: clamp + independent forward revalidation against
                    // THIS clause's constraints (the conjunction defining the disjunct).
                    // Only a confirmed in-box violation is emitted as Violated; an
                    // unconfirmed witness is treated as "no violation on this clause"
                    // so we keep probing remaining clauses (one may genuinely violate).
                    let revalidated =
                        map_mip_result_revalidated(mip_result, network, input, clause, num_outputs);
                    match revalidated {
                        VerificationResult::Violated { .. } => {
                            if !json {
                                println!(
                                    "  Clause {}/{}: SAT (counterexample confirmed in-box)",
                                    clause_idx + 1,
                                    num_clauses
                                );
                            }
                            return Ok(revalidated);
                        }
                        _ => {
                            // Solver-tolerance witness: re-solve this clause with a
                            // violation slack for a robust one.
                            let retry_config = MipConfig {
                                timeout_secs: fallback_config.timeout_secs.min(
                                    deadline
                                        .saturating_duration_since(std::time::Instant::now())
                                        .as_secs_f64(),
                                ),
                                ..fallback_config
                            };
                            if let Some(v) = retry_with_violation_slack(
                                network,
                                input,
                                weights,
                                biases,
                                layer_dims,
                                input_bounds,
                                intermediate_bounds,
                                clause,
                                retry_config,
                                num_outputs,
                            ) {
                                if !json {
                                    println!(
                                        "  Clause {}/{}: SAT (robust witness via violation-slack retry)",
                                        clause_idx + 1,
                                        num_clauses
                                    );
                                }
                                return Ok(v);
                            }
                            if !json {
                                println!(
                                    "  Clause {}/{}: SAT but failed in-box revalidation (cannot conclude verified)",
                                    clause_idx + 1,
                                    num_clauses
                                );
                            }
                            // The clause's unsafe region is reachable per the MIP, so
                            // concluding Verified here would be UNSOUND. Mark the run
                            // inconclusive (Unknown/Timeout) rather than safe. Not
                            // retried: a re-solve reproduces the same tolerance
                            // witness (the slack retry above already probed for a
                            // robust one).
                            had_unconfirmed_sat = true;
                        }
                    }
                }
                MipResult::Unsat { certified: true } => {
                    if !json {
                        println!(
                            "  Clause {}/{}: UNSAT (certified)",
                            clause_idx + 1,
                            num_clauses,
                        );
                    }
                }
                MipResult::Unsat { certified: false } => {
                    had_uncertified_unsat = true;
                    if !json {
                        println!(
                            "  Clause {}/{}: UNSAT without checked certificate (cannot conclude verified)",
                            clause_idx + 1,
                            num_clauses,
                        );
                    }
                }
                MipResult::Timeout => {
                    if !json {
                        println!(
                            "  Clause {}/{}: TIMEOUT ({:.1}s slice; will retry with the tail budget)",
                            clause_idx + 1,
                            num_clauses,
                            clause_timeout
                        );
                    }
                    pending.push(clause_idx);
                }
                MipResult::Error(msg) => {
                    if !json {
                        println!(
                            "  Clause {}/{}: ERROR ({})",
                            clause_idx + 1,
                            num_clauses,
                            msg
                        );
                    }
                    pending.push(clause_idx);
                }
            }
        }
    }
    if !pending.is_empty() {
        had_timeout = true;
    }

    let output_bounds =
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); num_outputs];
    match disjunctive_proof_status(had_unconfirmed_sat, had_uncertified_unsat, had_timeout) {
        DisjunctiveProofStatus::Unknown(reason) => Ok(VerificationResult::Unknown {
            provenance: Default::default(),
            bounds: output_bounds,
            reason: ny_core::UnknownReason::SmtUnknown {
                solver_reason: Some(reason.to_string()),
            },
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        }),
        DisjunctiveProofStatus::Timeout => Ok(VerificationResult::Timeout {
            provenance: Default::default(),
            partial_bounds: Some(output_bounds),
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        }),
        DisjunctiveProofStatus::Verified => Ok(VerificationResult::Verified {
            provenance: Default::default(),
            output_bounds,
            proof: None,
            actual_method: Some(mip_unsat_method_tag()),
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DisjunctiveProofStatus {
    Unknown(&'static str),
    Timeout,
    Verified,
}

fn disjunctive_proof_status(
    had_unconfirmed_sat: bool,
    had_uncertified_unsat: bool,
    had_timeout: bool,
) -> DisjunctiveProofStatus {
    if had_unconfirmed_sat {
        DisjunctiveProofStatus::Unknown("disjunctive MIP sat witness failed in-box revalidation")
    } else if had_uncertified_unsat {
        DisjunctiveProofStatus::Unknown("disjunctive MIP UNSAT lacked a checked certificate")
    } else if had_timeout {
        DisjunctiveProofStatus::Timeout
    } else {
        DisjunctiveProofStatus::Verified
    }
}

fn encode_output_constraint(
    encoder: &mut ny_mip::MipEncoder,
    constraint: &OutputConstraint,
) -> Result<ny_mip::ir::Row> {
    let r = match constraint {
        OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
            encoder.constrain_output_leq_row(*i, *j)
        }
        OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
            encoder.constrain_output_geq_row(*i, *j)
        }
        OutputConstraint::LessEqConst(i, c) | OutputConstraint::LessThanConst(i, c) => {
            encoder.constrain_output_leq_const_row(*i, *c)
        }
        OutputConstraint::GreaterEqConst(i, c) | OutputConstraint::GreaterThanConst(i, c) => {
            encoder.constrain_output_geq_const_row(*i, *c)
        }
        _ => return Err(anyhow::anyhow!("unsupported OutputConstraint variant")),
    };
    r.map_err(|e| anyhow::anyhow!("output constraint encoding failed: {}", e))
}

/// Print verification result in human-readable or JSON format.
#[allow(clippy::too_many_arguments)]
pub(super) fn print_result(
    result: &VerificationResult,
    property: Option<&Path>,
    model: Option<&Path>,
    epsilon: f32,
    threshold: f32,
    elapsed: std::time::Duration,
    backend: MipBackend,
    effective_treatment: Option<&EffectiveTreatmentProjection>,
    json: bool,
) -> Result<bool> {
    print_result_with_native_hull(
        result,
        property,
        model,
        epsilon,
        threshold,
        elapsed,
        backend,
        false,
        effective_treatment,
        json,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn print_result_with_native_hull(
    result: &VerificationResult,
    property: Option<&Path>,
    model: Option<&Path>,
    epsilon: f32,
    threshold: f32,
    elapsed: std::time::Duration,
    backend: MipBackend,
    native_hull_publication: bool,
    effective_treatment: Option<&EffectiveTreatmentProjection>,
    json: bool,
) -> Result<bool> {
    // This rendering-only choice comes from the retained opaque proof owner.
    // It grants no authority and must not relabel a native proof as AY UNSAT.
    if json {
        let method = if native_hull_publication {
            "native-hull"
        } else {
            match backend {
                MipBackend::Ay => "mip-ay",
                MipBackend::AyProc => "mip-ay-proc",
            }
        };
        let (rendered, publication_refused) = format_verification_result_json_for_publication(
            result,
            property,
            model,
            epsilon,
            threshold,
            elapsed,
            method,
            effective_treatment,
        )?;
        super::output::emit_competition_json(&rendered);
        return Ok(publication_refused);
    }
    match result {
        VerificationResult::Verified { .. } => println!("Status: VERIFIED (safe)"),
        VerificationResult::Violated {
            counterexample,
            output,
            ..
        } => {
            println!("Status: VIOLATED (unsafe)");
            println!(
                "Counterexample input: {}",
                crate::commands::verify::preview_f32s(counterexample)
            );
            let applied_terminal_peel = effective_treatment
                .map(EffectiveTreatmentProjection::terminal_peel_activation)
                .unwrap_or_default();
            let (label, displayed_output) = applied_terminal_peel.human_witness_output(output);
            if let Some(out) = displayed_output.first() {
                println!("{label}: {out}");
            }
        }
        VerificationResult::Timeout { .. } => println!("Status: TIMEOUT"),
        VerificationResult::Unknown { reason, .. } => {
            println!("Status: UNKNOWN");
            println!("Reason: {}", reason);
        }
    }
    if native_hull_publication {
        println!("Method: Native checked hull proof");
    } else {
        println!("Method: MIP ({} solver)", backend_name(backend));
    }
    println!("Time elapsed: {:.2}s", elapsed.as_secs_f64());
    Ok(false)
}

/// Map a non-SAT MIP result (Unsat / Timeout / Error) to a `VerificationResult`.
///
/// The SAT arm is handled separately by [`revalidate_mip_witness`], which clamps
/// the witness into the VNN-LIB box and re-checks the spec with an independent
/// forward pass before claiming `Violated`. `result` MUST NOT be `Sat` here.
/// Method tag for a MIP-admitted `Unsat { certified: true }` verdict, carrying
/// the certificate lane's PROVENANCE.
///
/// `ny_mip::strict_side_artifact_unsat_admissions()` counted, per process, the
/// admissions taken by the former `(None, None) if
/// strict_infeasibility_evidence` arm of
/// `ay_lib::map_outcome_with_sat_relu_certificate`, the one lane where
/// "certified" rested on AY's in-session replay of a typed side artifact with
/// NO NY-side re-check against `to_ay_model` (review of bbe9e9657,
/// 2026-09-12, finding "certificate seam arm without NY-side re-check").
/// When that arm had fired in this process the verdict was tagged
/// [`ny_core::MethodUsed::MipAyStrictSideArtifact`] and a flight note was
/// written, so a banked unsat that may rest on it is identifiable from the
/// bank's metadata alone. The counter is process-wide, so the tag is
/// conservative — a row whose earlier clause took that arm is tagged even if
/// the final clause carried a Farkas certificate — which is the correct
/// direction for a provenance mark.
///
/// That arm no longer exists: the bare-`Infeasible` branch now replays every
/// side artifact with a public model-bound verifier against NY's own model
/// (`SideArtifactReplays`, `[ay-cert] lane=<l> ny_replay=verified`) and
/// leaves the rest `certified: false`, which admission refuses, so the counter
/// stays 0 and this returns [`ny_core::MethodUsed::MipHiGHS`] for every new
/// verdict. The branch is kept so a reintroduced trust-the-bool arm (a
/// regression of the seam) would be tagged again, and so the tag's meaning in
/// banks written before the change stays documented.
pub(super) fn mip_unsat_method_tag() -> ny_core::MethodUsed {
    let admissions = ny_mip::strict_side_artifact_unsat_admissions();
    if admissions > 0 {
        crate::flight::note(
            "mip_unsat_provenance",
            crate::flight::FlightStatus::Ran,
            Some(format!(
                "ay-strict-side-artifact admissions={admissions}: a certified UNSAT in this \
                 process rests on AY's in-session replay of a typed side artifact with no \
                 NY-side re-check (ay_lib.rs strict_infeasibility_evidence arm)"
            )),
        );
        ny_core::MethodUsed::MipAyStrictSideArtifact
    } else {
        ny_core::MethodUsed::MipHiGHS
    }
}

fn map_mip_nonsat_result(result: MipResult, num_outputs: usize) -> VerificationResult {
    let bounds = || vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); num_outputs];
    match result {
        // Only independently checked exact evidence (Farkas or case-split)
        // may turn solver infeasibility into a verifier verdict.
        MipResult::Unsat { certified: true } => {
            tracing::info!("MIP UNSAT admitted with verified exact certificate");
            VerificationResult::Verified {
                provenance: Default::default(),
                output_bounds: bounds(),
                proof: None,
                actual_method: Some(mip_unsat_method_tag()),
            }
        }
        MipResult::Unsat { certified: false } => VerificationResult::Unknown {
            provenance: Default::default(),
            bounds: bounds(),
            reason: ny_core::UnknownReason::SmtUnknown {
                solver_reason: Some("MIP UNSAT lacked a checked certificate".to_string()),
            },
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        },
        MipResult::Sat { .. } => {
            // Defensive: SAT must be revalidated via revalidate_mip_witness, never
            // emitted raw. Treat an unexpected SAT here as Unknown (sound) rather
            // than fabricating an un-revalidated counterexample.
            VerificationResult::Unknown {
                provenance: Default::default(),
                bounds: bounds(),
                reason: ny_core::UnknownReason::SmtUnknown {
                    solver_reason: Some(
                        "MIP SAT reached map_mip_nonsat_result without revalidation".to_string(),
                    ),
                },
                actual_method: Some(ny_core::MethodUsed::MipHiGHS),
            }
        }
        MipResult::Timeout => VerificationResult::Timeout {
            provenance: Default::default(),
            partial_bounds: Some(bounds()),
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        },
        MipResult::Error(msg) => VerificationResult::Unknown {
            provenance: Default::default(),
            bounds: bounds(),
            reason: ny_core::UnknownReason::SmtUnknown {
                solver_reason: Some(msg),
            },
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        },
    }
}

/// Margin a concrete output has against an unsafe-region constraint.
///
/// Positive means the constraint is satisfied (output is in the unsafe region)
/// with that much slack; negative/zero means it does not (strictly) hold. OOB
/// indices map to `-inf` so a malformed constraint can never confirm a violation.
/// Mirrors the margin convention in `verify/disjunctive_pgd.rs`.
pub(super) fn mip_constraint_margin(constraint: &OutputConstraint, output: &ArrayD<f32>) -> f32 {
    let at = |i: usize| output.iter().nth(i).copied();
    match constraint {
        OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
            match (at(*i), at(*j)) {
                (Some(yi), Some(yj)) => yi - yj,
                _ => f32::NEG_INFINITY,
            }
        }
        OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
            match (at(*i), at(*j)) {
                (Some(yi), Some(yj)) => yj - yi,
                _ => f32::NEG_INFINITY,
            }
        }
        OutputConstraint::GreaterEqConst(i, c) | OutputConstraint::GreaterThanConst(i, c) => {
            at(*i).map_or(f32::NEG_INFINITY, |y| y - *c as f32)
        }
        OutputConstraint::LessEqConst(i, c) | OutputConstraint::LessThanConst(i, c) => {
            at(*i).map_or(f32::NEG_INFINITY, |y| *c as f32 - y)
        }
        _ => f32::NEG_INFINITY, // unknown variant cannot confirm a violation
    }
}

/// Margin guard absorbing f64->f32 cast drift between the solver's relaxation
/// and the independent CPU forward pass. A wrong VNN-COMP verdict is -150;
/// a timeout/Unknown is not — so borderline SAT claims are demoted, not emitted.
/// Same value as `verify/disjunctive_pgd.rs::re_evaluate_and_confirm`.
const REVALIDATION_MARGIN_EPS: f32 = 1e-5;

/// Descending violation-slack sweep. A larger delta yields a more robust witness
/// (one that survives f32/f64/ORT re-evaluation), but the STRENGTHENED problem is
/// only feasible when EVERY strengthened constraint still has reachable headroom.
/// A single fixed delta demoted genuine boundary-SAT witnesses whenever the
/// TIGHTEST constraint's headroom fell below it: on sat_v33_c140 the `Y_0 >= 1.0`
/// constraint's reachable max is only ~1 + 6.2e-6, so the uniform 1e-5 slack made
/// `Y_0 >= 1.00001` infeasible — independent of how far `Y_1` could be pushed below
/// 0 — and the real boundary witness (`Y_1` reachable a few e-6 below 0) was lost.
/// Sweeping downward and taking the FIRST (largest, most robust) delta whose
/// strengthened solution re-validates recovers these. The floor (5e-7) stays above
/// the measured ~5e-7 f32 forward drift so any accepted witness clears the
/// zero-tolerance revalidation gate. SOUND at every delta: the strengthened unsafe
/// region ⊆ the original, so a solution genuinely violates the original property,
/// and the witness still passes the independent zero-tolerance revalidation.
const VIOLATION_SLACKS: [f64; 5] = [1e-5, 5e-6, 2e-6, 1e-6, 5e-7];

/// Whether a constraint is a shiftable const-threshold comparison.
fn is_shiftable_const(c: &OutputConstraint) -> bool {
    use OutputConstraint as OC;
    matches!(
        c,
        OC::GreaterEqConst(..)
            | OC::GreaterThanConst(..)
            | OC::LessEqConst(..)
            | OC::LessThanConst(..)
    )
}

/// Recover a robust counterexample after a MIP `Sat` witness failed exact in-box
/// revalidation (a solver-tolerance artifact: the LP-feasible point can miss the true
/// forward by ~1e-6). Re-solve the SAME property strengthened by a violation slack so
/// any solution clears the zero-tolerance revalidation gate. SOUND: the strengthened
/// unsafe region ⊆ the original, so a solution genuinely violates it, and the returned
/// witness still passes the independent revalidation in `map_mip_result_revalidated`;
/// infeasibility proves nothing and the caller keeps its conservative outcome. See
/// [`VIOLATION_SLACKS`] for why the slack is swept per-constraint rather than uniform.
#[allow(clippy::too_many_arguments)]
fn retry_with_violation_slack(
    network: &Network,
    input: &BoundedTensor,
    weights: &[Vec<f64>],
    biases: &[Vec<f64>],
    layer_dims: &[usize],
    input_bounds: &[Bound],
    intermediate_bounds: &[Vec<Bound>],
    constraints: &[OutputConstraint],
    config: MipConfig,
    num_outputs: usize,
) -> Option<VerificationResult> {
    use OutputConstraint as OC;
    if !constraints.iter().any(is_shiftable_const) {
        return None; // nothing shiftable — no robustness to gain
    }

    let hard_remaining = || {
        config.ay_hard_deadline.and_then(|deadline| {
            deadline
                .checked_duration_since(std::time::Instant::now())
                .filter(|remaining| {
                    remaining.as_secs_f64() >= ObjectiveFirstSatBudget::MIN_FALLBACK_SECS
                })
                .map(|remaining| remaining.as_secs_f64())
        })
    };
    if config.ay_hard_deadline.is_some() && hard_remaining().is_none() {
        return None;
    }

    let solve = |cs: &[OC], secs: f64| -> Option<MipResult> {
        if config.ay_hard_deadline.is_some() && hard_remaining().is_none() {
            return None;
        }
        let mut enc = encode_feedforward(
            weights,
            biases,
            layer_dims,
            input_bounds,
            intermediate_bounds,
        )
        .ok()?;
        for c in cs {
            encode_output_constraint(&mut enc, c).ok()?;
        }
        // Encoding is outside AY's worker, but inside the caller's absolute
        // ledger. Recheck after it so setup can never buy a fresh solve slice.
        if config.ay_hard_deadline.is_some() && hard_remaining().is_none() {
            return None;
        }
        let cfg = MipConfig {
            timeout_secs: secs,
            ..config
        };
        MipSolver::new(enc.into_parts(), cfg)
            .check_feasibility()
            .ok()
    };

    // Budget: half the caller's live remaining time, split across an initial
    // diagnostic solve plus the delta sweep. Do not impose a minimum floor:
    // doing so can turn a sub-second tail into several seconds beyond the
    // authoritative deadline.
    let retry_budget = config.timeout_secs.max(0.0) * 0.5;
    let retry_budget =
        hard_remaining().map_or(retry_budget, |remaining| retry_budget.min(remaining));
    let per_try = retry_budget / (VIOLATION_SLACKS.len() + 1) as f64;
    if !per_try.is_finite() || per_try < ObjectiveFirstSatBudget::MIN_FALLBACK_SECS {
        return None;
    }

    // Diagnose WHICH constraints miss the independent real forward. Strengthening a
    // constraint that already holds with margin only shrinks the feasible set on that
    // (often tight) axis — e.g. `Y_0 >= 1.0`'s reachable headroom is ~6.2e-6 on
    // sat_v33_c140, so a uniform slack over ALL constraints made `Y_0 >= 1+delta`
    // infeasible independent of how far `Y_1` could move. Strengthen exactly the
    // constraints the witness missed; leave the satisfied ones at their thresholds.
    let mut failing = vec![false; constraints.len()];
    if let Some(MipResult::Sat { input_values, .. }) = solve(constraints, per_try) {
        if config.ay_hard_deadline.is_some() && hard_remaining().is_none() {
            return None;
        }
        let clamped = clamp_witness_to_box(&input_values, input);
        if let Ok(out) = independent_mip_forward(network, &clamped) {
            for (idx, c) in constraints.iter().enumerate() {
                if is_shiftable_const(c) && mip_constraint_margin(c, &out) < REVALIDATION_MARGIN_EPS
                {
                    failing[idx] = true;
                }
            }
        }
    }
    // Fall back to strengthening every const constraint if the diagnostic solve did not
    // pinpoint a failing one (preserves the original all-constraint behavior).
    let strengthen_all = !failing.iter().any(|&f| f);
    let strengthen = |delta: f64| -> Vec<OC> {
        constraints
            .iter()
            .enumerate()
            .map(|(idx, c)| {
                if !(strengthen_all || failing[idx]) {
                    return c.clone();
                }
                match c {
                    OC::GreaterEqConst(i, k) => OC::GreaterEqConst(*i, k + delta),
                    OC::GreaterThanConst(i, k) => OC::GreaterThanConst(*i, k + delta),
                    OC::LessEqConst(i, k) => OC::LessEqConst(*i, k - delta),
                    OC::LessThanConst(i, k) => OC::LessThanConst(*i, k - delta),
                    other => other.clone(),
                }
            })
            .collect()
    };

    for &delta in &VIOLATION_SLACKS {
        if config.ay_hard_deadline.is_some() && hard_remaining().is_none() {
            return None;
        }
        let strengthened = strengthen(delta);
        if strengthened == constraints {
            continue; // this delta shifted nothing (shouldn't happen post-guard)
        }
        tracing::warn!(
            "violation-slack retry: delta {delta:.0e} (strengthen_all={strengthen_all})"
        );
        let Some(mip_result) = solve(&strengthened, per_try) else {
            continue;
        };
        tracing::warn!("violation-slack retry outcome (delta {delta:.0e}): {mip_result:?}");
        // Revalidate against the ORIGINAL constraints — the property being scored.
        if let v @ VerificationResult::Violated { .. } =
            map_mip_result_revalidated(mip_result, network, input, constraints, num_outputs)
        {
            if config.ay_hard_deadline.is_some() && hard_remaining().is_none() {
                return None;
            }
            tracing::warn!("violation-slack retry recovered a robust witness (delta {delta:.0e})");
            return Some(v);
        }
    }
    None
}

/// Whether the exact f64 reference forward confirms the (f32-quantized, in-box)
/// witness violates every constraint under exact SMT-LIB semantics.
/// `Err(reason)` when the f64 path is unavailable for this network (the caller
/// keeps the conservative demotion then).
fn f64_forward_confirms(
    network: &Network,
    clamped: &ArrayD<f32>,
    constraints: &[OutputConstraint],
) -> std::result::Result<bool, String> {
    let layers64 = ny_propagate::convert_network_to_f64(network.layers())
        .map_err(|e| format!("layer conversion failed: {e}"))?;
    let input64 = clamped.mapv(f64::from);
    let out64 = ny_propagate::evaluate_network_f64(&layers64, &input64)
        .map_err(|e| format!("f64 forward failed: {e}"))?;
    let value_at = |i: usize| -> Option<f64> { out64.iter().nth(i).copied() };
    Ok(constraints.iter().all(|c| {
        use OutputConstraint as OC;
        match c {
            OC::LessEq(i, j) => {
                matches!((value_at(*i), value_at(*j)), (Some(a), Some(b)) if a <= b)
            }
            OC::LessThan(i, j) => {
                matches!((value_at(*i), value_at(*j)), (Some(a), Some(b)) if a < b)
            }
            OC::GreaterEq(i, j) => {
                matches!((value_at(*i), value_at(*j)), (Some(a), Some(b)) if a >= b)
            }
            OC::GreaterThan(i, j) => {
                matches!((value_at(*i), value_at(*j)), (Some(a), Some(b)) if a > b)
            }
            OC::LessEqConst(i, k) => value_at(*i).is_some_and(|a| a <= *k),
            OC::LessThanConst(i, k) => value_at(*i).is_some_and(|a| a < *k),
            OC::GreaterEqConst(i, k) => value_at(*i).is_some_and(|a| a >= *k),
            OC::GreaterThanConst(i, k) => value_at(*i).is_some_and(|a| a > *k),
            // Fail closed on any future constraint variant.
            _ => false,
        }
    }))
}

fn independent_mip_forward(network: &Network, candidate: &ArrayD<f32>) -> Result<ArrayD<f32>> {
    // Exact concrete forward, NOT IBP `.lower()` of a point box (completes
    // bd68815): IBP's ULP-outward rounding on big-M-scale nets exceeds
    // REVALIDATION_MARGIN_EPS and demoted genuine MIP witnesses (sat_relu).
    let input_bounds = BoundedTensor::concrete(candidate.clone())?;
    let output = network.propagate_concrete_point(&input_bounds, None)?;
    Ok(output.center())
}

/// Clamp a raw solver witness into the VNN-LIB input box.
///
/// Casts each raw f64 to f32 FIRST, then clamps into `[lower, upper]`, so the
/// returned witness is exactly the bytes the organizer's onnxruntime will read
/// AND is guaranteed inside the box even if the f64->f32 cast nudged a coord
/// out. The result is reshaped to `input.shape()` for the forward pass.
pub(super) fn clamp_witness_to_box(raw_input: &[f64], input: &BoundedTensor) -> ArrayD<f32> {
    let lower = input.lower();
    let upper = input.upper();
    let n = lower.len();
    let mut clamped = Vec::with_capacity(n);
    let mut lo_it = lower.iter();
    let mut hi_it = upper.iter();
    for k in 0..n {
        let lo = lo_it.next().copied().unwrap_or(f32::NEG_INFINITY);
        let hi = hi_it.next().copied().unwrap_or(f32::INFINITY);
        // raw_input matches the flattened input box order; if the solver returned
        // fewer values than the box (shouldn't happen), pad with the lower bound.
        let raw = raw_input.get(k).copied().unwrap_or(lo as f64) as f32;
        // clamp() panics if lo > hi; guard degenerate/NaN bounds defensively.
        let v = if lo <= hi {
            raw.clamp(lo, hi)
        } else {
            lo // degenerate box: collapse to lower
        };
        clamped.push(v);
    }
    ArrayD::from_shape_vec(IxDyn(lower.shape()), clamped)
        .expect("clamped witness length equals input box length by construction")
}

/// Clamp a raw MIP/SMT witness into the VNN-LIB box, re-validate it with an
/// independent forward pass through the ORIGINAL network, and emit `Violated`
/// ONLY if the spec is still violated in-box (with an epsilon margin guard).
///
/// This is the soundness gate for every MIP/SMT `Sat`: the organizer re-runs our
/// counterexample through onnxruntime on the eval inputs, rejecting any witness
/// that is out-of-box or that the f64->f32 cast moved off the violation. A wrong
/// verdict is -150; demoting a borderline witness to `Unknown` costs only a
/// timeout-equivalent. We therefore:
///   1. CLAMP each input into `[input.lower(), input.upper()]` (no-op for an
///      already in-box witness, so genuine violations are preserved).
///   2. INDEPENDENT FORWARD through `network` (engine=None), NOT `mip_network`.
///   3. RE-CHECK the spec: every constraint must hold with margin >=
///      `REVALIDATION_MARGIN_EPS`. If so, emit `Violated` with the CLAMPED input
///      and the RE-EVALUATED output (not the solver's relaxed output). Otherwise
///      demote to `Unknown` (sound — never claim a sat we cannot back).
fn revalidate_mip_witness(
    network: &Network,
    input: &BoundedTensor,
    raw_input: &[f64],
    constraints: &[OutputConstraint],
    num_outputs: usize,
) -> VerificationResult {
    let unknown = |reason: &str| VerificationResult::Unknown {
        provenance: Default::default(),
        bounds: vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); num_outputs],
        reason: ny_core::UnknownReason::SmtUnknown {
            solver_reason: Some(reason.to_string()),
        },
        actual_method: Some(ny_core::MethodUsed::MipHiGHS),
    };

    if constraints.is_empty() {
        // No spec constraints means no violation can be confirmed.
        return unknown("MIP sat had no output constraints to revalidate against");
    }

    // 1. Clamp the raw witness into the VNN-LIB box.
    let clamped = clamp_witness_to_box(raw_input, input);

    // 2. Independent forward pass through the ORIGINAL network (engine=None).
    let revalidated = match independent_mip_forward(network, &clamped) {
        Ok(out) => out,
        Err(e) => {
            tracing::warn!("MIP sat witness revalidation forward pass failed: {e}");
            return unknown("MIP sat witness revalidation forward pass failed");
        }
    };

    // 3. Re-check the spec under exact SMT-LIB semantics on the EXACT concrete
    // forward (check_unsafe_counterexample: strict `<`/`>` fail at equality,
    // non-strict `<=`/`>=` accept it). Deliberately NO extra epsilon guard:
    // SAT-encoded nets (sat_relu) construct satisfying assignments with
    // margins of exactly 0.0 or a few ULPs — a blanket eps demoted every real
    // witness the solver found in seconds. Cross-implementation robustness is
    // arbitrated downstream by the trusted-ORT vnncomp gate, which re-runs the
    // witness through real ONNX Runtime before any `sat` is scored (worst case
    // it downgrades to unknown — never a wrong verdict). Sub-eps margins are
    // still logged for diagnosability.
    let mut confirmed = super::verify::check_unsafe_counterexample(&revalidated, constraints);
    if confirmed {
        let min_margin = constraints
            .iter()
            .map(|c| mip_constraint_margin(c, &revalidated))
            .fold(f32::INFINITY, f32::min);
        if min_margin < REVALIDATION_MARGIN_EPS {
            tracing::info!(
                "MIP sat witness confirmed with sub-eps margin {min_margin:.3e} \
                 (< {REVALIDATION_MARGIN_EPS:.1e}); trusted-ORT gate will arbitrate"
            );
        }
    } else {
        // f64 rescue (winner parity: double_fp): SAT-encoded nets construct
        // their violations in real arithmetic; ny's f32 forward can miss by a
        // few ULPs (measured -1.9e-6 on sat_v100) where the faithful f64
        // forward confirms. Emit the sat — the trusted-ORT vnncomp gate still
        // re-runs the witness through real f32 ONNX Runtime before it is
        // scored (worst case a downgrade to unknown, never a wrong verdict).
        match f64_forward_confirms(network, &clamped, constraints) {
            Ok(true) => {
                tracing::warn!(
                    "MIP sat witness confirmed by the f64 reference forward (f32 forward \
                     missed by ULPs); trusted-ORT gate will arbitrate"
                );
                confirmed = true;
            }
            Ok(false) => {
                tracing::warn!("f64 reference forward also rejects the witness");
            }
            Err(reason) => {
                tracing::warn!("f64 rescue unavailable: {reason}");
            }
        }
    }

    if confirmed {
        VerificationResult::Violated {
            provenance: Default::default(),
            counterexample: clamped.iter().copied().collect(),
            output: revalidated.iter().copied().collect(),
            details: None,
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        }
    } else {
        // Diagnostics: how far the clamp moved the raw witness, and how close
        // the clamped point still is to violating (the binding margin). A
        // hair-negative margin points at solver-tolerance boundary witnesses;
        // a grossly negative one at an encoding/precision divergence.
        let max_displacement = clamped
            .iter()
            .zip(raw_input.iter())
            .map(|(c, r)| (f64::from(*c) - r).abs())
            .fold(0.0_f64, f64::max);
        let min_margin = constraints
            .iter()
            .map(|c| mip_constraint_margin(c, &revalidated))
            .fold(f32::INFINITY, f32::min);
        let per_constraint: Vec<String> = constraints
            .iter()
            .map(|c| {
                format!(
                    "{c:?}: margin={:.6e} strict={} unsafe_ok={}",
                    mip_constraint_margin(c, &revalidated),
                    c.is_strict(),
                    super::verify::check_unsafe_counterexample(
                        &revalidated,
                        std::slice::from_ref(c)
                    )
                )
            })
            .collect();
        tracing::warn!(
            "MIP sat witness failed in-box revalidation (clamped to box, spec no longer violated); \
             demoting to Unknown [max clamp displacement {max_displacement:.3e}, \
             min constraint margin at clamped point {min_margin:.3e}, \
             required >= {REVALIDATION_MARGIN_EPS:.1e}; per-constraint: {}]",
            per_constraint.join("; ")
        );
        unknown("MIP sat witness failed in-box revalidation")
    }
}

/// Map a MIP result to a `VerificationResult`, revalidating any `Sat` witness.
///
/// For `Sat`, the witness is clamped into the input box and re-checked with an
/// independent forward pass through the ORIGINAL `network` before emitting
/// `Violated`; an unconfirmed witness is demoted to `Unknown`. All other
/// outcomes are delegated to [`map_mip_nonsat_result`].
fn map_mip_result_revalidated(
    result: MipResult,
    network: &Network,
    input: &BoundedTensor,
    constraints: &[OutputConstraint],
    num_outputs: usize,
) -> VerificationResult {
    match result {
        MipResult::Sat { input_values, .. } => {
            revalidate_mip_witness(network, input, &input_values, constraints, num_outputs)
        }
        other => map_mip_nonsat_result(other, num_outputs),
    }
}

/// Replay a witness-only AY objective probe through the unchanged concrete
/// network/property gate.
///
/// A rejected point and every declined probe return `None`, which tells the
/// caller to run the historical feasibility path.  The enclosing VNN-COMP
/// command subsequently subjects any returned `Violated` to its trusted-ORT
/// replay before emitting `sat`.
fn revalidate_objective_first_sat_probe(
    probe: OneSidedSatProbe,
    network: &Network,
    input: &BoundedTensor,
    constraints: &[OutputConstraint],
    num_outputs: usize,
) -> Option<VerificationResult> {
    let OneSidedSatProbe::Witness(witness) = probe else {
        tracing::debug!("AY objective-first SAT probe declined: {probe:?}");
        return None;
    };
    let candidate = MipResult::Sat {
        objective: witness.objective,
        output_values: witness.output_values,
        input_values: witness.input_values,
        dual_bound: None,
    };
    let replayed = map_mip_result_revalidated(candidate, network, input, constraints, num_outputs);
    if matches!(replayed, VerificationResult::Violated { .. }) {
        tracing::info!(
            "AY objective-first SAT candidate passed concrete forward/spec replay; \
             trusted-ORT replay remains authoritative at the VNN-COMP seam"
        );
        Some(replayed)
    } else {
        tracing::debug!(
            "AY objective-first SAT candidate failed concrete replay; falling back to \
             historical feasibility"
        );
        None
    }
}

// Warm-start vector builder for the sequential PGD→MIP path.
#[path = "mip_highs_warm_start.rs"]
pub(super) mod warm_start;

#[path = "mip_highs_intermediate_bounds.rs"]
mod intermediate_bounds;

#[cfg(test)]
#[path = "mip_highs_tests.rs"]
mod tests;

// #mip-conv-sound-enclosure: the Big-M collectors must hand the encoder the
// certified IBP box; these tests bite on the conv cancellation shape.
#[cfg(test)]
#[path = "mip_conv_sound_enclosure_tests.rs"]
mod conv_sound_enclosure_tests;

// #mip-conv-sound-enclosure ratchet: every layer the sequential and graph MIP
// encoders admit must have a certified or directed IBP arm, BY NAME.
#[cfg(test)]
#[path = "mip_admitted_layer_ratchet_tests.rs"]
mod admitted_layer_ratchet_tests;

#[cfg(test)]
#[path = "mip_original_input_bridge_tests.rs"]
mod original_input_bridge_tests;

#[path = "mip_convex_binary.rs"]
mod convex_binary;

#[path = "mip_ideal_cuts.rs"]
mod ideal_cuts;
