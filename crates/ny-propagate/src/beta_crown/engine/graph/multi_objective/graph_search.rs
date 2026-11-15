// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #graph-lane-search (ENGINE_PROGRAM §2 M7): the graph β-CROWN
//! multi-objective lane's adapter for the injected UNTRUSTED batched α/β
//! search (`bab.graph_search: cpu_f32`; dark by default).
//!
//! # Where it runs
//!
//! * ROOT (`evaluate_root`, after every root objective phase and before the
//!   root verdict): the lane's certified root objective bounds are the
//!   INCUMBENT; the unresolved rows go to the backend as one unsplit domain.
//! * WAVES (`verify.rs`, the shared-executor branch, after the wave's children
//!   are bounded and BEFORE the containment fence, the replay tails and the
//!   queue fold): at most [`WAVE_MAX_DOMAINS`] open children per wave, the
//!   ones closest to closing first, go to the backend as ONE batch — each with
//!   its ReLU split literals, its own certified boxes and input box and its
//!   unresolved rows; the child's lane-certified bounds are its incumbent.
//!
//! # The pattern (M5a / V2), per domain
//!
//! 1. LOWER once per run: V2's `GraphSearchNet` lowering of the lane's own
//!    `GraphNetwork` (`crate::network::graph_search_lowering`), SELF-CHECKED
//!    against the lane's point forward (G1). A refusal or a mismatch refuses
//!    the phase.
//! 2. SEARCH, untrusted, on the registered backend (the same registry as V2,
//!    `network::graph_root_search`; ny-cli registers `ny_bbe::CpuF32Backend`).
//! 3. INGEST (the gate): a hinted slope reaches the lane's state only on a
//!    ReLU site of the lowering, only at a width equal to the certified
//!    pre-activation box's, only on neurons UNSTABLE in that CERTIFIED box,
//!    only when finite, and clamped into `[0, 1]`; a hinted multiplier reaches
//!    it only on the domain's own ReLU split literal, only when finite, and
//!    clamped to `≥ 0`. Everything else keeps the lane's value.
//! 4. CERTIFY ONCE with the lane's EXISTING evaluator, on the CPU (no engine:
//!    the device never certifies): the spec-guided CROWN backward
//!    (`SpecCrownRequest`) over the root's certified boxes, or the β-CROWN
//!    multi-objective pass (`propagate_multi_objective_with_beta_and_cache`)
//!    over the child's history, certified boxes and input box.
//! 5. KEEP ONLY IF STRICTLY BETTER, per row: a row's LOWER bound is raised to
//!    the candidate's certified lower when that is strictly larger; nothing
//!    else moves. At the root both ends of both intervals are certified, so a
//!    DISJOINT pair means one of two certified evaluations of the same value
//!    is wrong and nothing says which: the phase publishes nothing, prints an
//!    unconditional `[bbe-alarm]` and the root DOWNGRADES to `unknown`. An
//!    INVERTED candidate row (`lower > upper`, an empty enclosure) is the same
//!    alarm, never a raise of its lower alone; a NaN row is never read. A
//!    child's upper end carries no certificate for its sub-region
//!    (`multi_objective/shared.rs`), so a child row whose raised lower passes
//!    its old upper lifts that upper to the new lower (never lowers one) and
//!    is not an alarm.
//!
//! # Soundness
//!
//! Every number this module publishes is the output of the lane's own
//! certified evaluator at a state whose every searched parameter is finite and
//! inside its valid domain (any `α ∈ [0, 1]` is a ReLU lower slope; any
//! `β ≥ 0` is a Lagrange multiplier of the split `s z ≥ 0`), and it only ever
//! RAISES a lower bound that the lane itself already held, by `max`. A search
//! bug costs a missed improvement (+10), never a false bound: garbage hints
//! are refused or clamped
//! (`graph_lane_search_fault_injection_never_loosens_and_never_falsely_tightens`),
//! and a backend that PANICS is contained to its phase, which then refuses
//! (`backend_panicked`) and leaves the lane's own bounds.
//! Dark (`off`), no line of this module runs.
//!
//! # Observables
//!
//! `[bbe] phase=graph_bab stage=root …` once per armed root and
//! `[bbe] phase=graph_bab stage=wave …` per armed wave (the first 16, then
//! every 64th), each ending `graph-lane-search/v1`; every search number on
//! them is UNTRUSTED text, every `*_lb` is certified. An armed run without a
//! `phase=graph_bab` line is VOID.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_core::search::{
    GraphBabRequest, GraphDomainHint, GraphDomainRequest, GraphDomainSplit, SearchBackend,
    TensorBox,
};
use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::branching::GraphNeuronConstraint;
use crate::beta_crown::domain::{
    GraphCrownContext, MultiObjectiveGraphBabDomain, MultiObjectiveTargets,
};
use crate::beta_crown::engine::domain_results::MultiObjectiveGraphDomainResult;
use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};
use crate::bounds::GraphAlphaState;
use crate::network::graph_root_search::{
    graph_root_backend, record_graph_lane_search, GraphLaneSearchCounts,
};
use crate::network::graph_search_lowering::{
    lower_graph, self_check, AlphaSite, GraphLowering, SiteKind,
};
use crate::network::SpecCrownRequest;
use crate::{GraphNetwork, NETWORK_INPUT};

use super::super::super::BetaCrownVerifier;
use super::shared::{
    build_spec_matrix, prune_verified_multi_objective_targets, spec_bounds_to_vec,
};

/// Search iterations of the root (abc's `init_iteration`).
pub(super) const ROOT_SEARCH_ITERS: usize = 100;
/// Search iterations per BaB domain (abc's β-CROWN `iteration`).
pub(super) const DOMAIN_SEARCH_ITERS: usize = 20;
/// Open children one wave hands the backend (closest to closing first).
pub(super) const WAVE_MAX_DOMAINS: usize = 32;
/// Working set the backend may admit per call.
pub(super) const SEARCH_MEM_CAP: u64 = 1 << 30;
/// Share of the remaining root authority the root phase may spend, and its cap.
const ROOT_PHASE_SHARE: f64 = 0.25;
const ROOT_PHASE_MAX: Duration = Duration::from_secs(10);
/// Share of the remaining BaB wall one wave's phase may spend, and its cap.
const WAVE_PHASE_SHARE: f64 = 0.2;
const WAVE_PHASE_MAX: Duration = Duration::from_secs(5);
/// Share of a phase's slice the search may use (the rest is the certify reserve).
const SEARCH_SHARE: f64 = 0.5;
/// A slice shorter than this is not worth a lowering or a certification.
const MIN_SLICE: Duration = Duration::from_millis(20);
/// Largest graph (Σ certified box elements, input included) the phase lowers.
/// The search copy's conv gather tables grow as elements × taps and are built
/// before any backend admission, so a vgg-scale graph is refused up front
/// (fail closed: the lane's own bounds stand) instead of risking the host.
pub(super) const MAX_LOWERED_ELEMENTS: usize = 1 << 22;

/// Schema tag closing every `[bbe] phase=graph_bab` line.
const SCHEMA: &str = "graph-lane-search/v1";

/// Test-only record of the last certified root candidate on this thread: the
/// ingested state, the evaluated rows and their certified bounds, so a test can
/// re-derive the published number with an independent certified pass.
#[cfg(test)]
pub(super) struct RootCandidateRecord {
    pub(super) state: GraphAlphaState,
    pub(super) rows: Vec<usize>,
    pub(super) certified: Vec<(f32, f32)>,
    pub(super) ingest: IngestCounts,
}

#[cfg(test)]
thread_local! {
    pub(super) static LAST_ROOT_CANDIDATE: std::cell::RefCell<Option<RootCandidateRecord>> =
        const { std::cell::RefCell::new(None) };
}

/// Test-only record of one certified CHILD candidate: the child's split
/// history and input box (its sub-region), the evaluated rows and their
/// certified candidate bounds, so a test can check containment over the
/// sub-region at sampled points.
#[cfg(test)]
pub(super) struct ChildCandidateRecord {
    pub(super) history: crate::beta_crown::branching::GraphSplitHistory,
    pub(super) input: BoundedTensor,
    pub(super) rows: Vec<usize>,
    pub(super) certified: Vec<(f32, f32)>,
}

#[cfg(test)]
thread_local! {
    pub(super) static CERTIFIED_CHILDREN: std::cell::RefCell<Vec<ChildCandidateRecord>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

// ---------------------------------------------------------------------------
//  The keep rule (pure; pinned by the tests)
// ---------------------------------------------------------------------------

/// What merging a certified candidate into a ROOT incumbent did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RootMerge {
    /// The candidate's shape does not match: the incumbent stands.
    Ignored,
    /// No row strictly improved: the incumbent stands, bit for bit.
    NotBetter,
    /// `raised` rows took the candidate's certified lower.
    Kept { raised: usize },
    /// Row `row`'s two certified enclosures are DISJOINT, or the candidate's
    /// own certified enclosure of it is EMPTY (inverted): nothing was written.
    Alarm { row: usize },
}

/// THE ROOT KEEP RULE. `candidate[k]` is a certified enclosure of row
/// `rows[k]`, as is `incumbent[rows[k]]`. All-or-nothing on a disjoint pair
/// (checked for EVERY row before anything is written); otherwise each row's
/// lower becomes the candidate's when that is finite and strictly larger.
/// Upper ends never move.
///
/// A candidate row with a NaN end is not an enclosure and is never read (no
/// alarm, no raise). A candidate row that is INVERTED (`lower > upper`) is an
/// empty "enclosure" — disjoint from every interval — so it is an alarm, not a
/// raise: at the root both of its ends come from one certified pass, one of
/// them is wrong and nothing says which. Raising only its lower would publish
/// the ordered interval `[cl, iu]` and so walk straight past the lane's own
/// guard that an inverted interval grants no proof authority
/// (`BetaCrownConfig::domain_is_verified_for_mode`).
pub(super) fn merge_root_rows(
    incumbent: &mut [(f32, f32)],
    rows: &[usize],
    candidate: &[(f32, f32)],
) -> RootMerge {
    if candidate.len() != rows.len() || rows.iter().any(|&r| r >= incumbent.len()) {
        return RootMerge::Ignored;
    }
    for (&r, &(cl, cu)) in rows.iter().zip(candidate) {
        if cl.is_nan() || cu.is_nan() {
            continue;
        }
        let (il, iu) = incumbent[r];
        if cl > cu || cl > iu || cu < il {
            return RootMerge::Alarm { row: r };
        }
    }
    let mut raised = 0usize;
    for (&r, &(cl, cu)) in rows.iter().zip(candidate) {
        if cl.is_finite() && !cu.is_nan() && cl > incumbent[r].0 {
            incumbent[r].0 = cl;
            raised += 1;
        }
    }
    if raised == 0 {
        RootMerge::NotBetter
    } else {
        RootMerge::Kept { raised }
    }
}

/// THE CHILD KEEP RULE. Only the LOWER end of a BaB child's interval is
/// certified for its sub-region, so only lowers are compared: each row's lower
/// becomes the candidate's when that is finite and strictly larger, and an
/// upper below the new lower is lifted to it (never lowered). Returns the new
/// full vector and the number of rows raised, or `None` when nothing strictly
/// improved (or the shapes disagree) — the incumbent then stands bit for bit.
///
/// The lift is a DELIBERATE departure from the lane's inverted-interval guard
/// (`BetaCrownConfig::domain_is_verified_for_mode` refuses `lower > upper`
/// because it cannot tell which end is wrong). On a β-constrained child it CAN
/// tell: the upper end carries no certificate for the sub-region
/// (`multi_objective/shared.rs` `#violdrop`: 2366 of 2367 vit domains read
/// `upper < threshold` on an instance recorded `unsat`), while the lower is
/// the output of the lane's own certified β-CROWN pass. So an inversion there
/// is not evidence against the lower. The ROOT keeps the guard
/// (`merge_root_rows`): both of its ends are certified.
pub(super) fn merge_child_rows(
    incumbent: &[(f32, f32)],
    rows: &[usize],
    candidate: &[(f32, f32)],
) -> Option<(Vec<(f32, f32)>, usize)> {
    if candidate.len() != rows.len() || rows.iter().any(|&r| r >= incumbent.len()) {
        return None;
    }
    let mut out = incumbent.to_vec();
    let mut raised = 0usize;
    for (&r, &(cl, _)) in rows.iter().zip(candidate) {
        if cl.is_finite() && cl > out[r].0 {
            out[r] = (cl, out[r].1.max(cl));
            raised += 1;
        }
    }
    (raised > 0).then_some((out, raised))
}

// ---------------------------------------------------------------------------
//  The ingest gate
// ---------------------------------------------------------------------------

/// What an ingest wrote.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct IngestCounts {
    /// Values written (clamped slopes / multipliers).
    pub(super) written: usize,
    /// Non-finite values refused (the lane's value stands).
    pub(super) refused: usize,
    /// Hint entries that matched no site / literal of the lane.
    pub(super) unmatched: usize,
}

impl IngestCounts {
    fn add(&mut self, other: Self) {
        self.written += other.written;
        self.refused += other.refused;
        self.unmatched += other.unmatched;
    }
}

/// The certified pre-activation box of a lowered ReLU site.
fn site_pre_box<'a>(
    site: &AlphaSite,
    input: &'a BoundedTensor,
    pre_of: &impl Fn(&str) -> Option<&'a BoundedTensor>,
) -> Option<&'a BoundedTensor> {
    if site.input_node == NETWORK_INPUT {
        Some(input)
    } else {
        pre_of(&site.input_node)
    }
}

/// THE SLOPE INGEST GATE. Writes the hinted ReLU slopes into `state`'s LOWER
/// path: only on a ReLU site of the lowering, only when the hint's width
/// equals the certified pre-activation box's (a channel-only entry of the
/// state is replaced by a full-width one first), only on neurons UNSTABLE in
/// that CERTIFIED box, only finite values, clamped into `[0, 1]`. Every other
/// value of `state` is left as it was.
pub(super) fn ingest_relu_slopes<'a>(
    state: &mut GraphAlphaState,
    sites: &[AlphaSite],
    hint_relu: &[(usize, Vec<f32>)],
    input: &'a BoundedTensor,
    pre_of: impl Fn(&str) -> Option<&'a BoundedTensor>,
) -> IngestCounts {
    let mut counts = IngestCounts::default();
    for (op, slopes) in hint_relu {
        let Some(site) = sites
            .iter()
            .find(|s| s.op == *op && s.kind == SiteKind::Relu)
        else {
            counts.unmatched += 1;
            continue;
        };
        let Some(pre) = site_pre_box(site, input, &pre_of) else {
            counts.unmatched += 1;
            continue;
        };
        if pre.len() != slopes.len() {
            counts.unmatched += 1;
            continue;
        }
        let full = state.relu_len(&site.node) == Some(pre.len())
            && !state.spatial_shapes.contains_key(&site.node);
        if !full {
            // Transactional: `add_relu_node` runs every fallible step before
            // its first insert, so a refusal leaves the entry exactly as it
            // was. Only a SUCCESSFUL full-width entry may drop the old
            // channel-only shape — dropping it first would leave a length-C
            // slope vector that the backward reads as full width.
            if state.add_relu_node(&site.node, pre, false).is_err() {
                counts.unmatched += 1;
                continue;
            }
            state.spatial_shapes.remove(&site.node);
        }
        let Some((lower, _upper)) = state.relu_alpha_pair_mut(&site.node) else {
            counts.unmatched += 1;
            continue;
        };
        if lower.len() != slopes.len() {
            counts.unmatched += 1;
            continue;
        }
        for (j, ((&l, &u), &v)) in pre
            .lower()
            .iter()
            .zip(pre.upper().iter())
            .zip(slopes)
            .enumerate()
        {
            if !(l < 0.0 && u > 0.0) {
                continue;
            }
            if v.is_finite() {
                lower[j] = v.clamp(0.0, 1.0);
                counts.written += 1;
            } else {
                counts.refused += 1;
            }
        }
    }
    counts
}

/// THE MULTIPLIER INGEST GATE. `hinted[k]` is the multiplier of
/// `constraints[k]` (the request's literals were built in this order): written
/// only on that literal's own β entry, only finite, clamped to `≥ 0`.
pub(super) fn ingest_split_multipliers(
    beta: &mut GraphBetaState,
    constraints: &[GraphNeuronConstraint],
    hinted: &[f32],
) -> IngestCounts {
    let mut counts = IngestCounts {
        unmatched: hinted.len().abs_diff(constraints.len()),
        ..IngestCounts::default()
    };
    for (constraint, &v) in constraints.iter().zip(hinted) {
        if !v.is_finite() {
            counts.refused += 1;
            continue;
        }
        match beta.entry_mut(constraint.node_name(), constraint.neuron_idx()) {
            Some(entry) => {
                entry.set_value(v.max(0.0));
                counts.written += 1;
            }
            None => counts.unmatched += 1,
        }
    }
    counts
}

// ---------------------------------------------------------------------------
//  Shared plumbing
// ---------------------------------------------------------------------------

fn flat_f64(values: &ndarray::ArrayD<f32>) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

/// The certified box of `node` as a search reference box for tensor `tensor`.
fn tensor_box(tensor: usize, b: &BoundedTensor) -> TensorBox {
    TensorBox {
        tensor,
        l: flat_f64(b.lower()),
        u: flat_f64(b.upper()),
    }
}

/// A phase's wall: `(search deadline, certify deadline)`, or `None` when the
/// slice is too short to be worth anything.
fn phase_slice(
    now: Instant,
    authority: Option<Instant>,
    share: f64,
    cap: Duration,
) -> Option<(Instant, Instant)> {
    let remaining = authority.map_or(cap, |d| d.saturating_duration_since(now));
    let slice = cap.min(remaining.mul_f64(share));
    (slice >= MIN_SLICE).then(|| (now + slice.mul_f64(SEARCH_SHARE), now + slice))
}

/// Lower `graph` over `node_bounds` and self-check the copy (G1). Returns the
/// lowering and the number of (point, node) pairs compared.
pub(super) fn lower_and_check(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
) -> std::result::Result<(GraphLowering, usize), String> {
    let elements = node_bounds
        .values()
        .map(BoundedTensor::len)
        .fold(input.len(), usize::saturating_add);
    if elements > MAX_LOWERED_ELEMENTS {
        return Err(format!(
            "graph_too_large_for_search_copy elements={elements} cap={MAX_LOWERED_ELEMENTS}"
        ));
    }
    let order = graph.exec_order().map_err(|e| format!("exec_order: {e}"))?;
    let lowering = lower_graph(graph, order, input, node_bounds)?;
    if !lowering.sites.iter().any(|s| s.kind == SiteKind::Relu) {
        return Err("the lowered graph has no ReLU site".into());
    }
    let check = self_check(&lowering, graph, input, None).map_err(|e| format!("G1: {e}"))?;
    Ok((lowering, check.compared))
}

/// The lowering's ReLU site of the lane node `relu_node`.
fn relu_site<'a>(lowering: &'a GraphLowering, relu_node: &str) -> Option<&'a AlphaSite> {
    lowering
        .sites
        .iter()
        .find(|s| s.kind == SiteKind::Relu && s.node == relu_node)
}

/// Run one UNTRUSTED backend call with a panic contained to it: `None` when
/// the backend panicked. A search bug may cost this phase (+10 at most); it
/// must never unwind through the lane, which would take the lane's own
/// verdict with it. `ny_bbe`'s single-worker path (the root's one domain)
/// runs its search on THIS thread, so a panic there is not caught for us.
fn search_contained<T>(call: impl FnOnce() -> T) -> Option<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)).ok()
}

/// Untrusted search text (formatted so it cannot be compared with zero).
fn untrusted_text(v: &ny_core::search::Untrusted<f64>) -> String {
    format!("{v:.6}")
}

/// Minimum lower bound over `rows` (certified numbers, telemetry).
fn min_lower(bounds: &[(f32, f32)], rows: &[usize]) -> f32 {
    rows.iter()
        .filter_map(|&r| bounds.get(r).map(|b| b.0))
        .fold(f32::INFINITY, f32::min)
}

// ---------------------------------------------------------------------------
//  ROOT
// ---------------------------------------------------------------------------

/// Everything the root phase reads.
pub(super) struct RootPhase<'a> {
    pub(super) verifier: &'a BetaCrownVerifier,
    pub(super) graph: &'a GraphNetwork,
    pub(super) input: &'a BoundedTensor,
    pub(super) objectives: &'a [Vec<f32>],
    pub(super) thresholds: &'a [f32],
    /// The root's certified intermediate boxes (what its objective pass read).
    pub(super) node_bounds: &'a HashMap<String, BoundedTensor>,
    /// The root's α (warm start of the search only).
    pub(super) incumbent_alpha: Option<&'a GraphAlphaState>,
    /// The root authority (the phase takes a share of what is left).
    pub(super) deadline: Option<Instant>,
}

/// What the root phase did.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RootPhaseOutcome {
    /// `Some(reason)` when the phase refused before publishing anything.
    pub(super) refused: Option<String>,
    /// The merge decision (`None` when nothing was certified).
    pub(super) merge: Option<RootMerge>,
}

impl RootPhaseOutcome {
    /// A disjoint certified pair was found: the root must not publish.
    pub(super) fn alarm_row(&self) -> Option<usize> {
        match self.merge {
            Some(RootMerge::Alarm { row }) => Some(row),
            _ => None,
        }
    }
}

/// Fields of the root line.
#[derive(Default)]
struct RootReport {
    backend: &'static str,
    rows: usize,
    open: usize,
    lowered_ops: usize,
    structural: usize,
    probed: usize,
    relu_sites: usize,
    selfcheck: usize,
    iters: usize,
    evals: usize,
    search_s: f64,
    search_stop: &'static str,
    search_init_slack: String,
    search_best_slack: String,
    ingest: IngestCounts,
    certify_s: f64,
    incumbent_min_lb: f32,
    published_min_lb: f32,
    rows_raised: usize,
    kept: bool,
    alarm: Option<usize>,
}

fn emit_root_line(report: &RootReport, refused: Option<&str>) {
    eprintln!(
        "[bbe] phase=graph_bab stage=root backend={} refused={} rows={} open={} lowered_ops={} \
         structural={} probed={} relu_sites={} selfcheck_nodes={} iters={} evals={} \
         search_s={:.3} search_stop={} search_init_slack={} search_best_slack={} \
         ingest_written={} ingest_refused={} unmatched={} certify_s={:.3} \
         incumbent_min_lb={:.6} published_min_lb={:.6} rows_raised={} kept={} alarm={} {SCHEMA}",
        if report.backend.is_empty() {
            "none"
        } else {
            report.backend
        },
        refused.unwrap_or("none"),
        report.rows,
        report.open,
        report.lowered_ops,
        report.structural,
        report.probed,
        report.relu_sites,
        report.selfcheck,
        report.iters,
        report.evals,
        report.search_s,
        if report.search_stop.is_empty() {
            "not_run"
        } else {
            report.search_stop
        },
        if report.search_init_slack.is_empty() {
            "nan"
        } else {
            &report.search_init_slack
        },
        if report.search_best_slack.is_empty() {
            "nan"
        } else {
            &report.search_best_slack
        },
        report.ingest.written,
        report.ingest.refused,
        report.ingest.unmatched,
        report.certify_s,
        report.incumbent_min_lb,
        report.published_min_lb,
        report.rows_raised,
        u8::from(report.kept),
        report
            .alarm
            .map_or_else(|| "none".to_string(), |r| r.to_string()),
    );
}

/// The ARMED root phase. `objective_bounds` is the lane's certified root
/// incumbent; only its unresolved rows' LOWER bounds can move, and only up.
pub(super) fn run_root_phase(
    phase: &RootPhase<'_>,
    objective_bounds: &mut [(f32, f32)],
) -> RootPhaseOutcome {
    let mut report = RootReport {
        rows: objective_bounds.len(),
        ..RootReport::default()
    };
    let outcome = root_phase_inner(phase, objective_bounds, &mut report);
    let refused = outcome.refused.as_deref();
    emit_root_line(&report, refused);
    if let Some(RootMerge::Alarm { row }) = outcome.merge {
        let (il, iu) = objective_bounds[row];
        eprintln!(
            "[bbe-alarm] phase=graph_bab stage=root DISJOINT_SOUNDNESS_ALARM row={row} \
             incumbent=[{il},{iu}] published=none verdict=unknown {SCHEMA}"
        );
    }
    let mut delta = GraphLaneSearchCounts {
        root_phases: 1,
        ..GraphLaneSearchCounts::default()
    };
    match outcome.merge {
        None => delta.root_refused = 1,
        Some(merge) => {
            delta.root_certified = 1;
            match merge {
                RootMerge::Kept { raised } => {
                    delta.root_kept = 1;
                    delta.rows_raised = raised;
                }
                RootMerge::Alarm { .. } => delta.alarms = 1,
                RootMerge::Ignored | RootMerge::NotBetter => {}
            }
        }
    }
    record_graph_lane_search(&delta);
    outcome
}

fn refuse(reason: impl Into<String>) -> RootPhaseOutcome {
    RootPhaseOutcome {
        refused: Some(reason.into()),
        merge: None,
    }
}

fn root_phase_inner(
    phase: &RootPhase<'_>,
    objective_bounds: &mut [(f32, f32)],
    report: &mut RootReport,
) -> RootPhaseOutcome {
    let open: Vec<usize> = objective_bounds
        .iter()
        .zip(phase.thresholds)
        .enumerate()
        .filter(|&(_, (&(lower, _), &t))| {
            lower.partial_cmp(&t) != Some(std::cmp::Ordering::Greater)
        })
        .map(|(r, _)| r)
        .collect();
    report.open = open.len();
    report.incumbent_min_lb = min_lower(objective_bounds, &open);
    report.published_min_lb = report.incumbent_min_lb;
    if open.is_empty() {
        return refuse("no_open_row");
    }
    if phase.objectives.len() != objective_bounds.len()
        || phase.thresholds.len() != objective_bounds.len()
    {
        return refuse("objective_shape");
    }
    let Some((search_deadline, certify_deadline)) = phase_slice(
        Instant::now(),
        phase.deadline,
        ROOT_PHASE_SHARE,
        ROOT_PHASE_MAX,
    ) else {
        return refuse("no_time");
    };
    let Some(backend) = graph_root_backend() else {
        return refuse("no_backend");
    };
    report.backend = backend.name();
    let (lowering, compared) = match lower_and_check(phase.graph, phase.input, phase.node_bounds) {
        Ok(lowered) => lowered,
        Err(reason) => return refuse(format!("lowering:{}", reason.replace(' ', "_"))),
    };
    report.lowered_ops = lowering.net.ops.len();
    report.structural = lowering.structural;
    report.probed = lowering.probed;
    report.relu_sites = lowering
        .sites
        .iter()
        .filter(|s| s.kind == SiteKind::Relu)
        .count();
    report.selfcheck = compared;

    // The request: one unsplit domain over the unresolved rows, warm-started at
    // the incumbent's full-width slopes.
    let init_relu: Vec<(usize, Vec<f32>)> = phase
        .incumbent_alpha
        .map(|alpha| {
            lowering
                .sites
                .iter()
                .filter(|s| s.kind == SiteKind::Relu)
                .filter_map(|s| {
                    let values = alpha.alpha(&s.node)?;
                    let pre = site_pre_box(s, phase.input, &|n: &str| phase.node_bounds.get(n))?;
                    (values.len() == pre.len()).then(|| (s.op, values.to_vec()))
                })
                .collect()
        })
        .unwrap_or_default();
    let open_objectives: Vec<Vec<f32>> =
        open.iter().map(|&r| phase.objectives[r].clone()).collect();
    let request = GraphBabRequest {
        lo: flat_f64(phase.input.lower()),
        hi: flat_f64(phase.input.upper()),
        boxes: lowering.boxes.clone(),
        rows: open_objectives
            .iter()
            .map(|row| row.iter().map(|&c| f64::from(c)).collect())
            .collect(),
        thresholds: open
            .iter()
            .map(|&r| f64::from(phase.thresholds[r]))
            .collect(),
        domains: vec![GraphDomainRequest {
            init_relu,
            ..GraphDomainRequest::default()
        }],
        iters: ROOT_SEARCH_ITERS,
        lr_alpha: f64::from(phase.verifier.config.alpha_config.learning_rate),
        lr_beta: f64::from(phase.verifier.config.beta_lr),
        deadline: Some(search_deadline),
        threads: 0,
        mem_cap_bytes: SEARCH_MEM_CAP,
    };
    let t_search = Instant::now();
    let outcome = search_contained(|| backend.graph_bab_batch(&lowering.net, &request));
    report.search_s = t_search.elapsed().as_secs_f64();
    let outcome = match outcome {
        Some(Ok(outcome)) => outcome,
        Some(Err(refusal)) => {
            return refuse(format!("backend:{}", refusal.to_string().replace(' ', "_")))
        }
        None => return refuse("backend_panicked"),
    };
    if let Some(Some(stats)) = outcome.stats.per_domain.first() {
        report.iters = stats.iters;
        report.evals = stats.evals;
        report.search_stop = stats.stop;
        report.search_init_slack = untrusted_text(&stats.init_slack);
        report.search_best_slack = untrusted_text(&stats.best_slack);
    }
    let hints = outcome.hints.into_hint().into_ingest();
    let Some(Some(hint)) = hints.into_iter().next() else {
        return refuse("no_hint");
    };

    // INGEST: the incumbent state with the hinted slopes written in.
    let mut candidate = phase
        .incumbent_alpha
        .map_or_else(GraphAlphaState::new, |alpha| {
            let mut state = alpha.clone_for_backward();
            state.clear_spec_slots();
            state
        });
    report.ingest = ingest_relu_slopes(
        &mut candidate,
        &lowering.sites,
        &hint.alpha.relu,
        phase.input,
        |n: &str| phase.node_bounds.get(n),
    );
    if report.ingest.written == 0 {
        return refuse("nothing_ingested");
    }
    if Instant::now() >= certify_deadline {
        return refuse("no_time_to_certify");
    }

    // CERTIFY ONCE: the lane's spec-guided CROWN backward, on the CPU.
    let Some(spec) = build_spec_matrix(&open_objectives) else {
        return refuse("spec_matrix");
    };
    let t_certify = Instant::now();
    let certified = SpecCrownRequest::new(phase.graph, phase.input, &spec, None)
        .node_bounds(phase.node_bounds)
        .alpha_state_opt(Some(&candidate))
        .deadline_opt(Some(certify_deadline))
        .truncate_after_opt(phase.verifier.config.crown_backward_layers)
        .run();
    report.certify_s = t_certify.elapsed().as_secs_f64();
    let certified = match certified {
        Ok(bounds) => spec_bounds_to_vec(&bounds),
        Err(error) => return refuse(format!("certify:{}", error.to_string().replace(' ', "_"))),
    };
    #[cfg(test)]
    LAST_ROOT_CANDIDATE.with(|cell| {
        *cell.borrow_mut() = Some(RootCandidateRecord {
            state: candidate.clone(),
            rows: open.clone(),
            certified: certified.clone(),
            ingest: report.ingest,
        });
    });

    // KEEP ONLY IF STRICTLY BETTER.
    let merge = merge_root_rows(objective_bounds, &open, &certified);
    match merge {
        RootMerge::Kept { raised } => {
            report.kept = true;
            report.rows_raised = raised;
            report.published_min_lb = min_lower(objective_bounds, &open);
        }
        RootMerge::Alarm { row } => report.alarm = Some(row),
        RootMerge::Ignored | RootMerge::NotBetter => {}
    }
    RootPhaseOutcome {
        refused: None,
        merge: Some(merge),
    }
}

// ---------------------------------------------------------------------------
//  WAVES
// ---------------------------------------------------------------------------

/// The armed wave adapter: the lowering (built once, over the ROOT's certified
/// boxes, so the batch boxes are supersets of every child's) and the backend.
pub(super) struct GraphSearchWave {
    lowering: GraphLowering,
    /// The root input box (every child carries its own; this is the batch's).
    lo: Vec<f64>,
    hi: Vec<f64>,
    /// `(tensor, lane node)` of every reference box a child replaces with its
    /// own certified box.
    box_nodes: Vec<(usize, String)>,
    backend: Arc<dyn SearchBackend>,
    armed_waves: usize,
}

/// Fields of one wave's line.
#[derive(Default)]
struct WaveReport {
    open: usize,
    selected: usize,
    searched: usize,
    search_refused: usize,
    late: usize,
    certified: usize,
    certify_err: usize,
    kept: usize,
    closed: usize,
    rows_raised: usize,
    ingest: IngestCounts,
    search_s: f64,
    certify_s: f64,
}

/// The children of a result, when it has any.
fn children_of(
    result: &MultiObjectiveGraphDomainResult,
) -> Option<&Vec<(MultiObjectiveGraphBabDomain, bool)>> {
    match result {
        MultiObjectiveGraphDomainResult::Children(children)
        | MultiObjectiveGraphDomainResult::ChildrenWithViolatedDrop(children) => Some(children),
        _ => None,
    }
}

fn children_of_mut(
    result: &mut MultiObjectiveGraphDomainResult,
) -> Option<&mut Vec<(MultiObjectiveGraphBabDomain, bool)>> {
    match result {
        MultiObjectiveGraphDomainResult::Children(children)
        | MultiObjectiveGraphDomainResult::ChildrenWithViolatedDrop(children) => Some(children),
        _ => None,
    }
}

/// The worst unresolved-row slack of an open child (larger = closer to
/// closing); `None` when it has no unresolved row or a NaN bound.
fn open_child_key(child: &MultiObjectiveGraphBabDomain, thresholds: &[f32]) -> Option<f32> {
    let mut worst = f32::INFINITY;
    let mut any = false;
    for ((&(lower, _), &verified), &t) in child
        .objective_bounds()
        .iter()
        .zip(child.verified())
        .zip(thresholds)
    {
        if verified {
            continue;
        }
        if lower.is_nan() {
            return None;
        }
        any = true;
        worst = worst.min(lower - t);
    }
    any.then_some(worst)
}

impl GraphSearchWave {
    /// Arm the wave adapter for one BaB run, or print why not (`None`).
    pub(super) fn arm(
        verifier: &BetaCrownVerifier,
        graph: &GraphNetwork,
        root: &MultiObjectiveGraphBabDomain,
        refusal: Option<&'static str>,
    ) -> Option<Self> {
        if !verifier.config.graph_search.is_armed() {
            return None;
        }
        let refuse_line = |reason: &str| {
            eprintln!(
                "[bbe] phase=graph_bab stage=wave wave=arm backend=none refused={reason} {SCHEMA}"
            );
            record_graph_lane_search(&GraphLaneSearchCounts {
                waves: 1,
                wave_refused: 1,
                ..GraphLaneSearchCounts::default()
            });
        };
        if let Some(reason) = refusal {
            refuse_line(reason);
            return None;
        }
        let Some(backend) = graph_root_backend() else {
            refuse_line("no_backend");
            return None;
        };
        let node_bounds: HashMap<String, BoundedTensor> = root
            .node_bounds()
            .iter()
            .map(|(name, bounds)| (name.clone(), (**bounds).clone()))
            .collect();
        let (lowering, _) = match lower_and_check(graph, root.input_bounds(), &node_bounds) {
            Ok(lowered) => lowered,
            Err(reason) => {
                refuse_line(&format!("lowering:{}", reason.replace(' ', "_")));
                return None;
            }
        };
        let mut box_nodes: Vec<(usize, String)> = lowering
            .tensor_of
            .iter()
            .filter(|&(_, &t)| t != 0 && lowering.boxes.iter().any(|b| b.tensor == t))
            .map(|(name, &t)| (t, name.clone()))
            .collect();
        box_nodes.sort();
        Some(Self {
            lowering,
            lo: flat_f64(root.input_bounds().lower()),
            hi: flat_f64(root.input_bounds().upper()),
            box_nodes,
            backend,
            armed_waves: 0,
        })
    }

    /// The domain request of `child`, or `None` when its history is not a pure
    /// set of ReLU-at-zero literals on lowered ReLU sites.
    fn domain_request(
        &self,
        child: &MultiObjectiveGraphBabDomain,
        thresholds: &[f32],
    ) -> Option<GraphDomainRequest> {
        let history = child.history();
        if !history.is_pure_relu_at_zero() {
            return None;
        }
        let mut splits = Vec::with_capacity(history.constraints.len());
        let mut init_beta = Vec::with_capacity(history.constraints.len());
        for constraint in &history.constraints {
            let site = relu_site(&self.lowering, constraint.node_name())?;
            let neuron = u32::try_from(constraint.neuron_idx()).ok()?;
            splits.push(GraphDomainSplit {
                op: site.op,
                neuron,
                active: constraint.is_active(),
            });
            init_beta.push(
                child
                    .beta_state()
                    .entry(constraint.node_name(), constraint.neuron_idx())
                    .map_or(0.0, |entry| entry.value()),
            );
        }
        let boxes = self
            .box_nodes
            .iter()
            .filter_map(|(t, name)| child.node_bounds().get(name).map(|b| tensor_box(*t, b)))
            .collect();
        let rows = child
            .verified()
            .iter()
            .zip(thresholds)
            .enumerate()
            .filter(|&(_, (&verified, _))| !verified)
            .map(|(r, _)| r)
            .collect();
        Some(GraphDomainRequest {
            splits,
            boxes,
            lo: flat_f64(child.input_bounds().lower()),
            hi: flat_f64(child.input_bounds().upper()),
            rows,
            init_relu: Vec::new(),
            init_beta,
        })
    }

    /// ONE certified evaluation of `child` at the ingested hint: the lane's
    /// β-CROWN multi-objective pass over the child's history, certified boxes
    /// and input box, on the CPU. Returns the evaluated rows and their
    /// certified candidate bounds (`None` = nothing to evaluate).
    fn certify_child(
        &self,
        verifier: &BetaCrownVerifier,
        graph: &GraphNetwork,
        child: &MultiObjectiveGraphBabDomain,
        hint: &GraphDomainHint,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        ingest: &mut IngestCounts,
    ) -> Result<Option<(Vec<usize>, Vec<(f32, f32)>)>> {
        let mut global = GraphAlphaState::new();
        ingest.add(ingest_relu_slopes(
            &mut global,
            &self.lowering.sites,
            &hint.alpha.relu,
            child.input_bounds(),
            |name: &str| child.node_bounds().get(name).map(|b| &**b),
        ));
        let alpha = GraphDomainAlphaState::from_global_alpha_state_for_input_split(&global);
        let mut beta = child.beta_state().clone();
        ingest.add(ingest_split_multipliers(
            &mut beta,
            &child.history().constraints,
            &hint.beta,
        ));
        let pruned =
            prune_verified_multi_objective_targets(objectives, thresholds, child.verified());
        if pruned.active_indices.is_empty() {
            return Ok(None);
        }
        let targets = MultiObjectiveTargets::new(
            &pruned.objectives,
            &pruned.thresholds,
            &pruned.verified_mask,
        );
        let seeds: Vec<Option<&CachedLinearBounds>> = vec![None; pruned.objectives.len()];
        let context = GraphCrownContext::new_with_node_bounds_map(
            child.history(),
            None,
            Some(child.node_bounds()),
            None,
        )
        .with_alpha(&alpha);
        let (candidate, _node_bounds, _caches) = verifier
            .propagate_multi_objective_with_beta_and_cache(
                graph,
                child.input_bounds(),
                &context,
                &beta,
                &targets,
                &seeds,
                false,
            )?;
        #[cfg(test)]
        CERTIFIED_CHILDREN.with(|cell| {
            cell.borrow_mut().push(ChildCandidateRecord {
                history: child.history().clone(),
                input: child.input_bounds().clone(),
                rows: pruned.active_indices.clone(),
                certified: candidate.clone(),
            });
        });
        Ok(Some((pruned.active_indices, candidate)))
    }

    /// Run the armed phase over one shared-executor wave's results, BEFORE the
    /// fence, the replay tails and the fold. Only open children's LOWER bounds
    /// can move (only up), and an `all_verified` flag only ever turns on.
    pub(super) fn run(
        &mut self,
        verifier: &BetaCrownVerifier,
        graph: &GraphNetwork,
        results: &mut [MultiObjectiveGraphDomainResult],
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        bab_deadline: Instant,
        wave: usize,
    ) {
        let index = self.armed_waves;
        self.armed_waves += 1;
        let mut report = WaveReport::default();
        let refused = self.run_inner(
            verifier,
            graph,
            results,
            objectives,
            thresholds,
            bab_deadline,
            &mut report,
        );
        if index < 16 || index.is_multiple_of(64) {
            eprintln!(
                "[bbe] phase=graph_bab stage=wave wave={wave} backend={} refused={} open={} \
                 selected={} searched={} search_refused={} late={} certified={} certify_err={} \
                 kept={} closed={} rows_raised={} ingest_written={} ingest_refused={} \
                 unmatched={} search_s={:.3} certify_s={:.3} {SCHEMA}",
                self.backend.name(),
                refused.unwrap_or("none"),
                report.open,
                report.selected,
                report.searched,
                report.search_refused,
                report.late,
                report.certified,
                report.certify_err,
                report.kept,
                report.closed,
                report.rows_raised,
                report.ingest.written,
                report.ingest.refused,
                report.ingest.unmatched,
                report.search_s,
                report.certify_s,
            );
        }
        record_graph_lane_search(&GraphLaneSearchCounts {
            waves: 1,
            wave_refused: usize::from(report.certified == 0),
            domains_searched: report.searched,
            domains_certified: report.certified,
            domains_kept: report.kept,
            domains_closed: report.closed,
            rows_raised: report.rows_raised,
            ..GraphLaneSearchCounts::default()
        });
    }

    fn run_inner(
        &self,
        verifier: &BetaCrownVerifier,
        graph: &GraphNetwork,
        results: &mut [MultiObjectiveGraphDomainResult],
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        bab_deadline: Instant,
        report: &mut WaveReport,
    ) -> Option<&'static str> {
        if results
            .iter()
            .any(|r| matches!(r, MultiObjectiveGraphDomainResult::DeadlineExpired))
        {
            return Some("wave_deadline");
        }
        // Open children, closest to closing first (ties: wave order).
        let mut open: Vec<(f32, usize, usize)> = Vec::new();
        for (ri, result) in results.iter().enumerate() {
            let Some(children) = children_of(result) else {
                continue;
            };
            for (ci, (child, all_verified)) in children.iter().enumerate() {
                if *all_verified {
                    continue;
                }
                if let Some(key) = open_child_key(child, thresholds) {
                    open.push((key, ri, ci));
                }
            }
        }
        report.open = open.len();
        if open.is_empty() {
            return Some("no_open_child");
        }
        open.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
                .then(a.2.cmp(&b.2))
        });
        let Some((search_deadline, certify_deadline)) = phase_slice(
            Instant::now(),
            Some(bab_deadline),
            WAVE_PHASE_SHARE,
            WAVE_PHASE_MAX,
        ) else {
            return Some("no_time");
        };
        let mut selected: Vec<(usize, usize)> = Vec::new();
        let mut domains: Vec<GraphDomainRequest> = Vec::new();
        for &(_, ri, ci) in &open {
            if domains.len() >= WAVE_MAX_DOMAINS {
                break;
            }
            let Some(children) = children_of(&results[ri]) else {
                continue;
            };
            if let Some(domain) = self.domain_request(&children[ci].0, thresholds) {
                selected.push((ri, ci));
                domains.push(domain);
            }
        }
        report.selected = domains.len();
        if domains.is_empty() {
            return Some("no_pure_relu_child");
        }
        let request = GraphBabRequest {
            lo: self.lo.clone(),
            hi: self.hi.clone(),
            boxes: self.lowering.boxes.clone(),
            rows: objectives
                .iter()
                .map(|row| row.iter().map(|&c| f64::from(c)).collect())
                .collect(),
            thresholds: thresholds.iter().map(|&t| f64::from(t)).collect(),
            domains,
            iters: DOMAIN_SEARCH_ITERS,
            lr_alpha: f64::from(verifier.config.alpha_lr),
            lr_beta: f64::from(verifier.config.beta_lr),
            deadline: Some(search_deadline),
            threads: 0,
            mem_cap_bytes: SEARCH_MEM_CAP,
        };
        let t_search = Instant::now();
        let outcome =
            search_contained(|| self.backend.graph_bab_batch(&self.lowering.net, &request));
        report.search_s = t_search.elapsed().as_secs_f64();
        let outcome = match outcome {
            Some(Ok(outcome)) => outcome,
            Some(Err(_)) => return Some("backend_refused"),
            None => return Some("backend_panicked"),
        };
        report.searched = outcome.stats.searched;
        report.search_refused = outcome.stats.refused;
        report.late = outcome.stats.late;
        let hints = outcome.hints.into_hint().into_ingest();
        let t_certify = Instant::now();
        for (&(ri, ci), hint) in selected.iter().zip(hints) {
            let Some(hint) = hint else {
                continue;
            };
            if Instant::now() >= certify_deadline {
                break;
            }
            let Some(children) = children_of(&results[ri]) else {
                continue;
            };
            let certified = self.certify_child(
                verifier,
                graph,
                &children[ci].0,
                &hint,
                objectives,
                thresholds,
                &mut report.ingest,
            );
            let (rows, candidate) = match certified {
                Ok(Some(pair)) => pair,
                Ok(None) => continue,
                Err(_) => {
                    report.certify_err += 1;
                    continue;
                }
            };
            report.certified += 1;
            let Some(children) = children_of_mut(&mut results[ri]) else {
                continue;
            };
            let (child, all_verified) = &mut children[ci];
            let Some((merged, raised)) =
                merge_child_rows(child.objective_bounds(), &rows, &candidate)
            else {
                continue;
            };
            // `update_bounds` recomputes verified flags and priority with the
            // lane's own rules, and changes nothing when it refuses.
            if child.update_bounds(merged, thresholds, false).is_err() {
                continue;
            }
            report.kept += 1;
            report.rows_raised += raised;
            if child.all_verified() && !*all_verified {
                *all_verified = true;
                report.closed += 1;
            }
        }
        report.certify_s = t_certify.elapsed().as_secs_f64();
        None
    }
}
