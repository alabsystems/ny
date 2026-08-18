// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Objective-conditioned projected-lambda search for the resident Cut-CROWN
//! shadow.
//!
//! This module is deliberately telemetry-only.  Every nonzero multiplier
//! snapshot is rebuilt through the existing call-local exact-certificate
//! authority and evaluated through the existing resident cut fold.  The best
//! completed observation has no conversion into a verifier bound, warm start,
//! cache, coefficient state, or verdict.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::time::Instant;

use ny_core::{NyError, ResidentCutShadowDisposition, ResidentCutShadowOutcome, Result};

use super::certified_cut_authority::{ResidentCutCallContext, ResidentCutSnapshotGenerations};
use super::certified_cut_shadow::ProductionResidentCutShadowRequest;
use super::{
    certified_coupling_facet_certificates_exact_with_deadline,
    combined_row_octahedron_with_deadline, ExactRelu2FacetCertificate,
};

/// The M2 gate is subordinate to the M1 resident-shadow gate.  Exact `"1"`
/// spelling is load-bearing so an invalid or child-only request remains dark.
const M2_PROJECTED_GATE: &str = "NY_CUT_CROWN_M2_PROJECTED";
const M1_RESIDENT_SHADOW_GATE: &str = "NY_CUT_CROWN_RESIDENT_SHADOW";

/// Keep proposal and resident-refold work tightly bounded.  With two pairs and
/// two facets per pair, seed ranking costs at most four snapshots.  Two
/// coordinate sweeps cost at most eight more snapshots.
const M2_MAX_TARGETS_SCANNED: usize = 4;
const M2_MAX_UNSTABLE_PER_TARGET: usize = 4;
const M2_MAX_PAIR_CANDIDATES: usize = 2;
const M2_MAX_FACETS_PER_PAIR: usize = 2;
const M2_MAX_LAMBDA: f32 = 4.0;
const M2_SEED_LAMBDA: f32 = 1.0;
const M2_COORDINATE_STEPS: [f32; 2] = [1.0, 0.5];
const M2_MAX_SNAPSHOT_ATTEMPTS: usize = M2_MAX_PAIR_CANDIDATES * M2_MAX_FACETS_PER_PAIR
    + M2_COORDINATE_STEPS.len() * M2_MAX_FACETS_PER_PAIR * 2;

pub(crate) fn production_resident_cut_m2_projected_enabled() -> bool {
    std::env::var(M1_RESIDENT_SHADOW_GATE).ok().as_deref() == Some("1")
        && std::env::var(M2_PROJECTED_GATE).ok().as_deref() == Some("1")
}

/// Return `None` before touching the request when the exact subordinate gate
/// is dark.  This is the M2 zero-work boundary; M1 remains independently
/// available under its parent gate.
pub(super) fn maybe_run_production_resident_cut_m2_projected(
    request: &ProductionResidentCutShadowRequest<'_>,
) -> Option<Result<ResidentCutShadowOutcome>> {
    if !production_resident_cut_m2_projected_enabled() {
        return None;
    }
    Some(run_production_resident_cut_m2_projected(request))
}

#[derive(Debug)]
struct ProjectedCandidate {
    target_activation: usize,
    ordered_neurons: [usize; 2],
    certificates: Vec<ExactRelu2FacetCertificate>,
}

#[derive(Clone, Copy, Debug)]
struct PairProposal {
    target_activation: usize,
    ordered_neurons: [usize; 2],
    structural_width: f64,
}

/// Build a bounded structural proposal set before any resident refold.  Width
/// is only a cheap prefilter; candidate/facet ordering used by the optimizer is
/// determined later from actual error-adjusted binding-row observations.
fn select_projected_candidates(
    request: &ProductionResidentCutShadowRequest<'_>,
) -> Result<Vec<ProjectedCandidate>> {
    check_deadline(request.deadline, "before M2 candidate prefilter")?;
    if request.relu_names.is_empty()
        || request.relu_names.len() != request.beta_signed.len()
        || request.relu_names.len() != request.node_abs.len()
        || request.frontier_abs.len() != request.segments.len()
    {
        return Err(NyError::InvalidSpec(
            "Cut-CROWN M2 has inconsistent resident decomposition shapes".into(),
        ));
    }

    let mut proposals = Vec::new();
    // The loop only ever leaves an iteration by falling through or by
    // returning an error, so the enumerate index is exactly the number of
    // targets already scanned; it doubles as the scan cap counter.
    for (target_activation, target_relu) in request.relu_names.iter().enumerate() {
        if target_activation >= M2_MAX_TARGETS_SCANNED {
            break;
        }
        check_deadline(request.deadline, "during M2 target prefilter")?;
        if request
            .relu_names
            .iter()
            .filter(|name| *name == target_relu)
            .count()
            != 1
        {
            return Err(NyError::InvalidSpec(
                "Cut-CROWN M2 target identity is not unique in the resident decomposition".into(),
            ));
        }
        let node = request.graph.node(target_relu).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Cut-CROWN M2 resident target '{target_relu}' is absent from the graph"
            ))
        })?;
        if !matches!(node.layer(), crate::layers::Layer::ReLU(_)) {
            return Err(NyError::InvalidSpec(format!(
                "Cut-CROWN M2 resident target '{target_relu}' is not an exact ReLU"
            )));
        }

        let pre_node = node.require_unary_input()?;
        let pre_bounds = request.node_bounds.get(pre_node).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Cut-CROWN M2 current bounds omit pre-activation '{pre_node}'"
            ))
        })?;
        let post_bounds = request.node_bounds.get(target_relu).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Cut-CROWN M2 current bounds omit target ReLU '{target_relu}'"
            ))
        })?;
        let pre = pre_bounds.flatten();
        let post = post_bounds.flatten();
        if pre.is_empty()
            || pre.len() != post.len()
            || request.beta_signed[target_activation].len() != pre.len()
            || request.node_abs[target_activation].len() != pre.len()
        {
            return Err(NyError::InvalidSpec(format!(
                "Cut-CROWN M2 target '{target_relu}' disagrees with resident activation width"
            )));
        }
        if pre
            .lower()
            .iter()
            .zip(pre.upper())
            .chain(post.lower().iter().zip(post.upper()))
            .any(|(&lower, &upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
        {
            return Err(NyError::NumericalInstability(format!(
                "Cut-CROWN M2 target '{target_relu}' has invalid current bounds"
            )));
        }

        let mut unstable = pre
            .lower()
            .iter()
            .zip(pre.upper())
            .enumerate()
            .filter_map(|(index, (&lower, &upper))| {
                let width = f64::from(upper) - f64::from(lower);
                (lower < 0.0 && upper > 0.0 && width.is_finite()).then_some((index, width))
            })
            .collect::<Vec<_>>();
        unstable.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        unstable.truncate(M2_MAX_UNSTABLE_PER_TARGET);
        for left in 0..unstable.len() {
            for right in (left + 1)..unstable.len() {
                let mut ordered_neurons = [unstable[left].0, unstable[right].0];
                ordered_neurons.sort_unstable();
                proposals.push(PairProposal {
                    target_activation,
                    ordered_neurons,
                    structural_width: unstable[left].1 + unstable[right].1,
                });
            }
        }
    }
    proposals.sort_by(|left, right| {
        right
            .structural_width
            .total_cmp(&left.structural_width)
            .then_with(|| left.target_activation.cmp(&right.target_activation))
            .then_with(|| left.ordered_neurons.cmp(&right.ordered_neurons))
    });
    proposals.truncate(M2_MAX_PAIR_CANDIDATES);

    let mut candidates = Vec::with_capacity(proposals.len());
    for proposal in proposals {
        check_deadline(request.deadline, "before M2 exact pair production")?;
        let target_relu = &request.relu_names[proposal.target_activation];
        let pre_node = request
            .graph
            .node(target_relu)
            .ok_or_else(|| {
                NyError::SoundnessRefusal(
                    "Cut-CROWN M2 target identity changed after prefilter".into(),
                )
            })?
            .require_unary_input()?;
        let support = combined_row_octahedron_with_deadline(
            request.graph,
            request.input,
            request.alpha_state,
            Some(request.node_bounds),
            pre_node,
            proposal.ordered_neurons[0],
            proposal.ordered_neurons[1],
            Some(request.engine),
            Some(request.deadline),
        )?;
        let mut certificates = certified_coupling_facet_certificates_exact_with_deadline(
            &support,
            Some(request.deadline),
        )?;
        if certificates.iter().any(|certificate| {
            let facet = certificate.facet();
            !facet.b.is_finite() || facet.a.iter().any(|value| !value.is_finite())
        }) {
            return Err(NyError::NumericalInstability(
                "Cut-CROWN M2 exact certificate contains non-finite data".into(),
            ));
        }

        // Exact certification establishes validity; this deterministic pass
        // only removes duplicate/degenerate work and caps the expensive actual
        // fold ranking.  It carries no objective score.
        let mut seen = HashSet::new();
        certificates.retain(|certificate| {
            let facet = certificate.facet();
            let key = [
                facet.a[0].to_bits(),
                facet.a[1].to_bits(),
                facet.a[2].to_bits(),
                facet.a[3].to_bits(),
                facet.b.to_bits(),
            ];
            facet.is_coupling() && facet.a.iter().any(|value| *value != 0.0) && seen.insert(key)
        });
        certificates.sort_by(|left, right| {
            facet_structural_strength(right)
                .total_cmp(&facet_structural_strength(left))
                .then_with(|| facet_bits(left).cmp(&facet_bits(right)))
        });
        certificates.truncate(M2_MAX_FACETS_PER_PAIR);
        if !certificates.is_empty() {
            candidates.push(ProjectedCandidate {
                target_activation: proposal.target_activation,
                ordered_neurons: proposal.ordered_neurons,
                certificates,
            });
        }
        check_deadline(request.deadline, "after M2 exact pair production")?;
    }
    if candidates.is_empty() {
        return Err(NyError::UnsupportedOp(
            "Cut-CROWN M2 found no bounded exact-certified k=2 candidates".into(),
        ));
    }
    Ok(candidates)
}

fn facet_structural_strength(certificate: &ExactRelu2FacetCertificate) -> f64 {
    let facet = certificate.facet();
    facet.a.iter().map(|value| f64::from(value.abs())).sum()
}

fn facet_bits(certificate: &ExactRelu2FacetCertificate) -> [u32; 5] {
    let facet = certificate.facet();
    [
        facet.a[0].to_bits(),
        facet.a[1].to_bits(),
        facet.a[2].to_bits(),
        facet.a[3].to_bits(),
        facet.b.to_bits(),
    ]
}

pub(super) fn run_production_resident_cut_m2_projected(
    request: &ProductionResidentCutShadowRequest<'_>,
) -> Result<ResidentCutShadowOutcome> {
    if request.binding_row >= request.seed.num_specs {
        return Err(NyError::InvalidSpec(
            "Cut-CROWN M2 binding row is outside the objective seed".into(),
        ));
    }
    let candidates = select_projected_candidates(request)?;
    let facet_counts = candidates
        .iter()
        .map(|candidate| candidate.certificates.len())
        .collect::<Vec<_>>();
    let search = projected_coordinate_search(
        &facet_counts,
        request.binding_row,
        request.deadline,
        |candidate_index, lambdas| {
            evaluate_projected_snapshot(request, &candidates[candidate_index], lambdas)
        },
    )?;
    let selected = &candidates[search.best.candidate_index];
    eprintln!(
        "{} target_activation={} ordered_neurons={:?}",
        search.telemetry.render(),
        selected.target_activation,
        selected.ordered_neurons,
    );
    Ok(search.best.outcome)
}

fn evaluate_projected_snapshot(
    request: &ProductionResidentCutShadowRequest<'_>,
    candidate: &ProjectedCandidate,
    lambdas: &[f32],
) -> Result<ResidentCutShadowOutcome> {
    check_deadline(
        request.deadline,
        "before M2 snapshot authority construction",
    )?;
    let row_lambdas = binding_row_lambda_matrix(
        request.seed.num_specs,
        request.binding_row,
        candidate.certificates.len(),
        lambdas,
    )?;
    let target_relu = request
        .relu_names
        .get(candidate.target_activation)
        .ok_or_else(|| {
            NyError::SoundnessRefusal(
                "Cut-CROWN M2 target activation moved after candidate selection".into(),
            )
        })?;
    let context = ResidentCutCallContext::new(
        ResidentCutSnapshotGenerations::initial(),
        request.graph,
        request.input,
        request.alpha_state,
        request.node_bounds,
        Some(request.engine),
        request.seed,
        target_relu,
        candidate.ordered_neurons,
        request.segments,
        request.relu_names,
        request.beta_signed,
        request.frontier_abs,
        request.node_abs,
        request.resident_input_lower,
        request.resident_input_upper,
        request.deadline,
    );
    let carrier = context
        .build_bound_carrier(&candidate.certificates, &row_lambdas)?
        .ok_or_else(|| {
            NyError::SoundnessRefusal("Cut-CROWN M2 projected snapshot is exactly all-zero".into())
        })?;
    context.run_backend_shadow(&carrier, request.gpu, request.binding_row)
}

fn validate_projected_lambdas(facet_count: usize, lambdas: &[f32]) -> Result<()> {
    if facet_count == 0 || facet_count > M2_MAX_FACETS_PER_PAIR || lambdas.len() != facet_count {
        return Err(NyError::InvalidSpec(
            "Cut-CROWN M2 projected lambda shape is outside the bounded facet policy".into(),
        ));
    }
    if let Some(value) = lambdas
        .iter()
        .copied()
        .find(|value| !value.is_finite() || *value < 0.0 || *value > M2_MAX_LAMBDA)
    {
        return Err(NyError::InvalidSpec(format!(
            "Cut-CROWN M2 lambda must be finite in [0,{M2_MAX_LAMBDA}], got {value}"
        )));
    }
    if lambdas.iter().all(|value| *value == 0.0) {
        return Err(NyError::InvalidSpec(
            "Cut-CROWN M2 does not dispatch an all-zero snapshot".into(),
        ));
    }
    Ok(())
}

fn binding_row_lambda_matrix(
    row_count: usize,
    binding_row: usize,
    facet_count: usize,
    lambdas: &[f32],
) -> Result<Vec<Vec<f32>>> {
    if row_count == 0 || binding_row >= row_count {
        return Err(NyError::InvalidSpec(
            "Cut-CROWN M2 binding row is outside the lower objective matrix".into(),
        ));
    }
    validate_projected_lambdas(facet_count, lambdas)?;
    let mut rows = vec![vec![0.0; facet_count]; row_count];
    rows[binding_row].copy_from_slice(lambdas);
    Ok(rows)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BaselineFingerprint {
    lower: Vec<u32>,
    upper: Vec<u32>,
}

impl BaselineFingerprint {
    fn from_outcome(outcome: &ResidentCutShadowOutcome) -> Result<Self> {
        let baseline = outcome.baseline();
        if baseline
            .lower_bounds
            .iter()
            .chain(&baseline.upper_bounds)
            .any(|value| !value.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "Cut-CROWN M2 baseline fingerprint contains a non-finite bound".into(),
            ));
        }
        Ok(Self {
            lower: baseline
                .lower_bounds
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            upper: baseline
                .upper_bounds
                .iter()
                .map(|value| value.to_bits())
                .collect(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum M2SearchStop {
    Completed,
    Exhausted,
    Deadline,
    BackendRefusal,
    IdentityMismatch,
    InvalidSnapshot,
}

#[derive(Clone, Debug)]
struct CompletedSnapshot {
    candidate_index: usize,
    lambdas: Vec<f32>,
    delta: f32,
    outcome: ResidentCutShadowOutcome,
}

#[derive(Clone, Debug)]
struct M2ProjectedTelemetry {
    candidate_pairs: usize,
    exact_facets: usize,
    snapshots_attempted: usize,
    snapshots_completed: usize,
    selected_candidate: usize,
    best_lambdas: Vec<f32>,
    best_delta: f32,
    stop: M2SearchStop,
}

impl M2ProjectedTelemetry {
    fn render(&self) -> String {
        format!(
            "[cut-crown-m2-projected] telemetry_only=true authority=false \
             candidate_pairs={} exact_facets={} snapshots_attempted={} \
             snapshots_completed={} max_snapshots={} selected_candidate={} \
             best_lambdas={:?} best_delta={} stop={:?}",
            self.candidate_pairs,
            self.exact_facets,
            self.snapshots_attempted,
            self.snapshots_completed,
            M2_MAX_SNAPSHOT_ATTEMPTS,
            self.selected_candidate,
            self.best_lambdas,
            self.best_delta,
            self.stop,
        )
    }
}

#[derive(Debug)]
struct M2SearchResult {
    best: CompletedSnapshot,
    telemetry: M2ProjectedTelemetry,
}

#[derive(Default)]
struct M2SearchState {
    baseline: Option<BaselineFingerprint>,
    best: Option<CompletedSnapshot>,
    attempted: usize,
    completed: usize,
    stop: Option<M2SearchStop>,
}

impl M2SearchState {
    fn accept(
        &mut self,
        candidate_index: usize,
        lambdas: &[f32],
        binding_row: usize,
        outcome: ResidentCutShadowOutcome,
    ) -> Result<f32> {
        if outcome.disposition() != ResidentCutShadowDisposition::Observed {
            self.stop = Some(M2SearchStop::BackendRefusal);
            return Err(NyError::SoundnessRefusal(format!(
                "Cut-CROWN M2 backend did not complete a snapshot: {:?}",
                outcome.disposition()
            )));
        }
        let observation = outcome.observation().ok_or_else(|| {
            self.stop = Some(M2SearchStop::InvalidSnapshot);
            NyError::SoundnessRefusal(
                "Cut-CROWN M2 observed disposition omitted its observation".into(),
            )
        })?;
        if observation.binding_row() != binding_row || !observation.delta().is_finite() {
            self.stop = Some(M2SearchStop::InvalidSnapshot);
            return Err(NyError::SoundnessRefusal(
                "Cut-CROWN M2 observation does not match the finite binding objective".into(),
            ));
        }
        let fingerprint = match BaselineFingerprint::from_outcome(&outcome) {
            Ok(fingerprint) => fingerprint,
            Err(error) => {
                self.stop = Some(M2SearchStop::InvalidSnapshot);
                return Err(error);
            }
        };
        if self
            .baseline
            .as_ref()
            .is_some_and(|baseline| baseline != &fingerprint)
        {
            self.stop = Some(M2SearchStop::IdentityMismatch);
            return Err(NyError::SoundnessRefusal(
                "Cut-CROWN M2 backend baseline identity changed between snapshots".into(),
            ));
        }
        self.baseline.get_or_insert(fingerprint);
        self.completed += 1;
        let delta = observation.delta();
        if self
            .best
            .as_ref()
            .is_none_or(|best| delta.total_cmp(&best.delta) == Ordering::Greater)
        {
            self.best = Some(CompletedSnapshot {
                candidate_index,
                lambdas: lambdas.to_vec(),
                delta,
                outcome,
            });
        }
        Ok(delta)
    }
}

/// Evaluate one complete nonzero snapshot.  A failed/late attempt never enters
/// `completed`; if an earlier snapshot completed, the caller retains that
/// telemetry atomically and stops.  With no completed snapshot, the error is
/// propagated and M2 publishes nothing.
fn evaluate_trial<E, N>(
    state: &mut M2SearchState,
    candidate_index: usize,
    lambdas: &[f32],
    binding_row: usize,
    deadline: Instant,
    evaluate: &mut E,
    now: &mut N,
) -> Result<Option<f32>>
where
    E: FnMut(usize, &[f32]) -> Result<ResidentCutShadowOutcome>,
    N: FnMut() -> Instant,
{
    if state.stop.is_some() {
        return Ok(None);
    }
    if now() >= deadline {
        state.stop = Some(M2SearchStop::Deadline);
        if state.best.is_some() {
            return Ok(None);
        }
        return Err(NyError::DeadlineExceeded(
            "Cut-CROWN M2 expired before its first completed snapshot".into(),
        ));
    }
    validate_projected_lambdas(lambdas.len(), lambdas)?;
    if state.attempted >= M2_MAX_SNAPSHOT_ATTEMPTS {
        state.stop = Some(M2SearchStop::Exhausted);
        return Ok(None);
    }
    state.attempted += 1;
    let outcome = match evaluate(candidate_index, lambdas) {
        Ok(outcome) => outcome,
        Err(error) => {
            state.stop = Some(if matches!(&error, NyError::DeadlineExceeded(_)) {
                M2SearchStop::Deadline
            } else {
                M2SearchStop::BackendRefusal
            });
            if state.best.is_some() {
                return Ok(None);
            }
            return Err(error);
        }
    };
    // A backend can return a syntactically complete outcome just after its
    // explicit child deadline. Recheck immediately before `accept`: a late
    // outcome is not a completed snapshot and can never replace an earlier
    // atomically retained observation.
    if now() >= deadline {
        state.stop = Some(M2SearchStop::Deadline);
        if state.best.is_some() {
            return Ok(None);
        }
        return Err(NyError::DeadlineExceeded(
            "Cut-CROWN M2 backend returned after the first snapshot deadline".into(),
        ));
    }
    match state.accept(candidate_index, lambdas, binding_row, outcome) {
        Ok(delta) => Ok(Some(delta)),
        Err(_) if state.best.is_some() => Ok(None),
        Err(error) => Err(error),
    }
}

/// Seed-rank every retained pair/facet by a real resident-fold delta, then run
/// a bounded projected coordinate search on the best pair.  Scores are the
/// backend's error-adjusted binding-row lower-bound improvements, not a
/// coefficient proxy.
fn projected_coordinate_search<E>(
    facet_counts: &[usize],
    binding_row: usize,
    deadline: Instant,
    evaluate: E,
) -> Result<M2SearchResult>
where
    E: FnMut(usize, &[f32]) -> Result<ResidentCutShadowOutcome>,
{
    projected_coordinate_search_with_clock(
        facet_counts,
        binding_row,
        deadline,
        evaluate,
        Instant::now,
    )
}

fn projected_coordinate_search_with_clock<E, N>(
    facet_counts: &[usize],
    binding_row: usize,
    deadline: Instant,
    mut evaluate: E,
    mut now: N,
) -> Result<M2SearchResult>
where
    E: FnMut(usize, &[f32]) -> Result<ResidentCutShadowOutcome>,
    N: FnMut() -> Instant,
{
    if facet_counts.is_empty()
        || facet_counts.len() > M2_MAX_PAIR_CANDIDATES
        || facet_counts
            .iter()
            .any(|count| *count == 0 || *count > M2_MAX_FACETS_PER_PAIR)
    {
        return Err(NyError::InvalidSpec(
            "Cut-CROWN M2 candidate plan is outside the bounded search policy".into(),
        ));
    }
    let mut state = M2SearchState::default();
    let mut facet_scores = facet_counts
        .iter()
        .map(|count| vec![f32::NEG_INFINITY; *count])
        .collect::<Vec<_>>();

    'seed: for (candidate_index, &facet_count) in facet_counts.iter().enumerate() {
        for facet_index in 0..facet_count {
            let mut lambdas = vec![0.0; facet_count];
            lambdas[facet_index] = M2_SEED_LAMBDA;
            let score = evaluate_trial(
                &mut state,
                candidate_index,
                &lambdas,
                binding_row,
                deadline,
                &mut evaluate,
                &mut now,
            )?;
            let Some(score) = score else {
                break 'seed;
            };
            facet_scores[candidate_index][facet_index] = score;
        }
    }

    if state.stop.is_none() {
        let selected_candidate = facet_scores
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                best_finite_score(left)
                    .total_cmp(&best_finite_score(right))
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .ok_or_else(|| {
                NyError::SoundnessRefusal(
                    "Cut-CROWN M2 seed ranking completed no finite candidate".into(),
                )
            })?;
        let mut facet_order = (0..facet_counts[selected_candidate]).collect::<Vec<_>>();
        facet_order.sort_by(|&left, &right| {
            facet_scores[selected_candidate][right]
                .total_cmp(&facet_scores[selected_candidate][left])
                .then_with(|| left.cmp(&right))
        });
        let initial_facet = facet_order[0];
        let mut current = vec![0.0; facet_counts[selected_candidate]];
        current[initial_facet] = M2_SEED_LAMBDA;
        let mut current_score = facet_scores[selected_candidate][initial_facet];

        'coordinate: for step in M2_COORDINATE_STEPS {
            for &facet_index in &facet_order {
                for direction in [1.0_f32, -1.0_f32] {
                    let mut trial = current.clone();
                    trial[facet_index] =
                        (trial[facet_index] + direction * step).clamp(0.0, M2_MAX_LAMBDA);
                    if trial == current || trial.iter().all(|value| *value == 0.0) {
                        continue;
                    }
                    let score = evaluate_trial(
                        &mut state,
                        selected_candidate,
                        &trial,
                        binding_row,
                        deadline,
                        &mut evaluate,
                        &mut now,
                    )?;
                    let Some(score) = score else {
                        break 'coordinate;
                    };
                    if score.total_cmp(&current_score) == Ordering::Greater {
                        current = trial;
                        current_score = score;
                    }
                }
            }
        }
    }

    let stop = state.stop.unwrap_or(M2SearchStop::Completed);
    let best = state.best.ok_or_else(|| {
        NyError::SoundnessRefusal(
            "Cut-CROWN M2 completed no identity-bound resident snapshot".into(),
        )
    })?;
    let telemetry = M2ProjectedTelemetry {
        candidate_pairs: facet_counts.len(),
        exact_facets: facet_counts.iter().sum(),
        snapshots_attempted: state.attempted,
        snapshots_completed: state.completed,
        selected_candidate: best.candidate_index,
        best_lambdas: best.lambdas.clone(),
        best_delta: best.delta,
        stop,
    };
    Ok(M2SearchResult { best, telemetry })
}

fn best_finite_score(scores: &[f32]) -> f32 {
    scores.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn check_deadline(deadline: Instant, stage: &str) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(format!(
            "Cut-CROWN M2 deadline exceeded {stage}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Duration;

    use ny_core::{GpuCrownResult, ResidentCutShadowObservation};
    use proptest::prelude::*;

    use super::*;

    fn observed(baseline: f32, delta: f32) -> ResidentCutShadowOutcome {
        let baseline_result = GpuCrownResult {
            lower_bounds: vec![baseline],
            upper_bounds: vec![3.0],
        };
        let observation = ResidentCutShadowObservation::try_new(0, baseline, baseline + delta)
            .expect("finite synthetic M2 observation");
        ResidentCutShadowOutcome::try_observed(baseline_result, observation)
            .expect("exact synthetic baseline binding")
    }

    fn observed_with_nonfinite_nonbinding_row(
        baseline: f32,
        delta: f32,
    ) -> ResidentCutShadowOutcome {
        let baseline_result = GpuCrownResult {
            lower_bounds: vec![baseline, f32::NAN],
            upper_bounds: vec![3.0, 3.0],
        };
        let observation = ResidentCutShadowObservation::try_new(0, baseline, baseline + delta)
            .expect("finite synthetic binding observation");
        ResidentCutShadowOutcome::try_observed(baseline_result, observation)
            .expect("the core wrapper binds only the declared row")
    }

    #[test]
    fn subordinate_gate_off_executes_zero_m2_work() {
        ny_test_utils::env::with_env_edits(|env| {
            env.remove(M1_RESIDENT_SHADOW_GATE);
            env.remove(M2_PROJECTED_GATE);
            let calls = AtomicUsize::new(0);
            let maybe_run = || {
                if production_resident_cut_m2_projected_enabled() {
                    calls.fetch_add(1, AtomicOrdering::SeqCst);
                }
            };
            maybe_run();
            assert_eq!(calls.load(AtomicOrdering::SeqCst), 0);

            env.set(M2_PROJECTED_GATE, "1");
            maybe_run();
            assert_eq!(
                calls.load(AtomicOrdering::SeqCst),
                0,
                "the child gate cannot arm M2 without its parent"
            );

            env.set(M1_RESIDENT_SHADOW_GATE, "1");
            maybe_run();
            assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);

            env.set(M2_PROJECTED_GATE, "true");
            maybe_run();
            assert_eq!(
                calls.load(AtomicOrdering::SeqCst),
                1,
                "only exact child spelling 1 may enter M2"
            );
        });
    }

    #[test]
    fn lambda_domain_rejects_nonfinite_negative_oversized_and_all_zero() {
        for invalid in [
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![-f32::from_bits(1)],
            vec![M2_MAX_LAMBDA + 1.0],
            vec![0.0],
        ] {
            assert!(validate_projected_lambdas(1, &invalid).is_err());
        }
        validate_projected_lambdas(2, &[0.0, M2_MAX_LAMBDA])
            .expect("closed finite nonnegative lambda domain");
    }

    proptest! {
        #[test]
        fn binding_matrix_is_row_local_for_every_admitted_shape(
            rows in 1usize..=64,
            binding_seed in 0usize..1024,
            first in 0u16..=4000,
            second in 0u16..=4000,
        ) {
            let binding = binding_seed % rows;
            let mut lambdas = [f32::from(first) / 1000.0, f32::from(second) / 1000.0];
            if lambdas == [0.0, 0.0] {
                lambdas[0] = 1.0;
            }
            let matrix = binding_row_lambda_matrix(rows, binding, 2, &lambdas)
                .expect("generated lambda matrix is within the M2 domain");
            prop_assert_eq!(&matrix[binding], &lambdas);
            for (row, values) in matrix.iter().enumerate() {
                if row != binding {
                    prop_assert_eq!(values, &[0.0, 0.0]);
                }
            }
        }
    }

    #[test]
    fn projected_search_climbs_a_bounded_coordinate_and_is_deterministic() {
        let deadline = Instant::now() + Duration::from_secs(30);
        let run = projected_coordinate_search(&[1], 0, deadline, |_, lambdas| {
            let lambda = lambdas[0];
            let delta = 4.0 - (lambda - 2.0) * (lambda - 2.0);
            Ok(observed(-3.0, delta))
        })
        .expect("bounded synthetic coordinate search");
        assert_eq!(run.best.lambdas, vec![2.0]);
        assert_eq!(run.best.delta, 4.0);
        assert!(run.telemetry.snapshots_attempted <= M2_MAX_SNAPSHOT_ATTEMPTS);
        assert_eq!(
            run.telemetry.snapshots_attempted,
            run.telemetry.snapshots_completed
        );
    }

    #[test]
    fn seed_ranking_uses_actual_binding_delta_across_pairs_and_facets() {
        let run = projected_coordinate_search(
            &[2, 2],
            0,
            Instant::now() + Duration::from_secs(30),
            |candidate, lambdas| {
                let delta = if candidate == 0 {
                    0.1 * lambdas.iter().sum::<f32>()
                } else {
                    2.0 * lambdas[0] + 3.0 * lambdas[1]
                };
                Ok(observed(-3.0, delta))
            },
        )
        .expect("actual-delta candidate ranking");
        assert_eq!(run.best.candidate_index, 1);
        assert!(
            run.best.delta >= 3.0,
            "the stronger second pair/facet must win"
        );
        assert_eq!(run.telemetry.selected_candidate, 1);
    }

    #[test]
    fn deadline_after_one_complete_snapshot_publishes_no_partial_snapshot() {
        let calls = AtomicUsize::new(0);
        let run = projected_coordinate_search(
            &[2],
            0,
            Instant::now() + Duration::from_secs(30),
            |_, _| {
                let call = calls.fetch_add(1, AtomicOrdering::SeqCst);
                if call == 0 {
                    Ok(observed(-3.0, 0.25))
                } else {
                    Err(NyError::DeadlineExceeded(
                        "synthetic mid-snapshot deadline".into(),
                    ))
                }
            },
        )
        .expect("the first complete snapshot is retained");
        assert_eq!(run.best.delta, 0.25);
        assert_eq!(run.telemetry.snapshots_attempted, 2);
        assert_eq!(run.telemetry.snapshots_completed, 1);
        assert_eq!(run.telemetry.stop, M2SearchStop::Deadline);
    }

    #[test]
    fn backend_returning_late_on_first_snapshot_publishes_nothing() {
        let before_deadline = Instant::now();
        let deadline = before_deadline + Duration::from_secs(30);
        let mut clock = [before_deadline, deadline].into_iter();
        let calls = AtomicUsize::new(0);

        let error = projected_coordinate_search_with_clock(
            &[1],
            0,
            deadline,
            |_, _| {
                calls.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(observed(-3.0, 100.0))
            },
            || clock.next().unwrap_or(deadline),
        )
        .expect_err("a late first backend result must not become an observation");

        assert!(error.is_deadline_exceeded());
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn backend_returning_late_later_retains_only_prior_complete_snapshot() {
        let before_deadline = Instant::now();
        let deadline = before_deadline + Duration::from_secs(30);
        let mut clock = [before_deadline, before_deadline, before_deadline, deadline].into_iter();
        let calls = AtomicUsize::new(0);

        let run = projected_coordinate_search_with_clock(
            &[2],
            0,
            deadline,
            |_, _| {
                let call = calls.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(observed(-3.0, if call == 0 { 0.25 } else { 100.0 }))
            },
            || clock.next().unwrap_or(deadline),
        )
        .expect("the first on-time snapshot remains atomically retained");

        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
        assert_eq!(run.best.delta, 0.25);
        assert_eq!(run.telemetry.snapshots_attempted, 2);
        assert_eq!(run.telemetry.snapshots_completed, 1);
        assert_eq!(run.telemetry.stop, M2SearchStop::Deadline);
    }

    #[test]
    fn baseline_identity_mismatch_retains_only_the_prior_complete_snapshot() {
        let calls = AtomicUsize::new(0);
        let run = projected_coordinate_search(
            &[2],
            0,
            Instant::now() + Duration::from_secs(30),
            |_, _| {
                let call = calls.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(if call == 0 {
                    observed(-3.0, 0.5)
                } else {
                    observed(-4.0, 10.0)
                })
            },
        )
        .expect("identity mismatch cannot erase the prior atomic observation");
        assert_eq!(run.best.delta, 0.5);
        assert_eq!(run.telemetry.snapshots_completed, 1);
        assert_eq!(run.telemetry.stop, M2SearchStop::IdentityMismatch);
    }

    #[test]
    fn nonfinite_baseline_component_stops_without_accepting_a_partial_snapshot() {
        let calls = AtomicUsize::new(0);
        let run = projected_coordinate_search(
            &[2],
            0,
            Instant::now() + Duration::from_secs(30),
            |_, _| {
                let call = calls.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(if call == 0 {
                    observed(-3.0, 0.5)
                } else {
                    observed_with_nonfinite_nonbinding_row(-3.0, 100.0)
                })
            },
        )
        .expect("the prior complete finite snapshot is retained");
        assert_eq!(run.best.delta, 0.5);
        assert_eq!(run.telemetry.snapshots_completed, 1);
        assert_eq!(run.telemetry.stop, M2SearchStop::InvalidSnapshot);
    }

    #[test]
    fn backend_failure_retains_the_best_completed_candidate_snapshot() {
        let calls = AtomicUsize::new(0);
        let run = projected_coordinate_search(
            &[2],
            0,
            Instant::now() + Duration::from_secs(30),
            |_, _| match calls.fetch_add(1, AtomicOrdering::SeqCst) {
                0 => Ok(observed(-3.0, 0.25)),
                1 => Ok(observed(-3.0, 0.75)),
                _ => Err(NyError::UnsupportedOp("synthetic backend refusal".into())),
            },
        )
        .expect("best complete snapshot survives a later backend refusal");
        assert_eq!(run.best.delta, 0.75);
        assert_eq!(run.best.lambdas, vec![0.0, 1.0]);
        assert_eq!(run.telemetry.snapshots_completed, 2);
        assert_eq!(run.telemetry.stop, M2SearchStop::BackendRefusal);
    }

    #[test]
    fn telemetry_is_explicitly_non_authoritative_and_reports_search_seams() {
        let telemetry = M2ProjectedTelemetry {
            candidate_pairs: 2,
            exact_facets: 4,
            snapshots_attempted: 7,
            snapshots_completed: 6,
            selected_candidate: 1,
            best_lambdas: vec![0.5, 1.0],
            best_delta: 0.125,
            stop: M2SearchStop::Deadline,
        };
        let rendered = telemetry.render();
        assert!(rendered.contains("telemetry_only=true authority=false"));
        assert!(rendered.contains("candidate_pairs=2 exact_facets=4"));
        assert!(rendered.contains("snapshots_attempted=7 snapshots_completed=6"));
        assert!(rendered.contains("best_lambdas=[0.5, 1.0]"));
        assert!(rendered.contains("stop=Deadline"));
    }
}
