// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Deadline-bounded CUDA SPSA for one multi-objective BaB child.
//!
//! Only full Standard propagations are verdict-bearing. The two-row CUDA calls
//! below are optimization probes; a refused, malformed, or late probe merely
//! stops optimization and returns the best already-completed Standard result.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_core::{
    GpuCrownBackward, GpuCrownSeed, NyError, Result, DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
};
use ny_tensor::BoundedTensor;

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::config::AdaptiveOptConfig;
use crate::beta_crown::domain::{GraphCrownContext, MultiObjectiveTargets};
use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};
use crate::GraphNetwork;

use super::super::propagation::batched::{prep_resnet_domain_ext, ResnetDomainPrep};
use super::{BetaCrownVerifier, MoCudaBetaSpsaFrontier, MultiObjectiveWarmStartResult};

const MAX_SPSA_UPDATES: usize = 3;
const SPSA_EPSILON: f32 = 1.0e-3;

struct CompletedStandard {
    margin: f32,
    bounds: Vec<(f32, f32)>,
    node_bounds: HashMap<String, Arc<BoundedTensor>>,
    beta: GraphBetaState,
}

struct ProbeBestSnapshot {
    score: f32,
    update: Option<usize>,
    beta: GraphBetaState,
}

struct ProbeLoopOutcome {
    best: ProbeBestSnapshot,
    attempted: usize,
    completed: usize,
}

impl ProbeBestSnapshot {
    fn consider_post_update(&mut self, score: f32, update: usize, beta: &GraphBetaState) -> bool {
        if !probe_candidate_improves(score, self.score) {
            return false;
        }
        self.score = score;
        self.update = Some(update);
        self.beta = beta.clone();
        true
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SpsaPhaseSummary {
    split_count: usize,
    beta_entries: usize,
    baseline_margin: f32,
    frontier: MoCudaBetaSpsaFrontier,
    frontier_admitted: bool,
    final_margin: Option<f32>,
    fixed_critical_row: Option<usize>,
    updates_attempted: usize,
    updates_completed: usize,
    selected_update: Option<usize>,
    final_admitted: bool,
    scalar_selected: bool,
}

fn option_value<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn frontier_value(frontier: MoCudaBetaSpsaFrontier) -> String {
    match frontier {
        MoCudaBetaSpsaFrontier::Empty => "none".to_string(),
        MoCudaBetaSpsaFrontier::Finite(value) => format!("{value:.6}"),
        MoCudaBetaSpsaFrontier::Invalid => "invalid".to_string(),
    }
}

fn spsa_phase_summary_line(summary: SpsaPhaseSummary, elapsed: Duration) -> String {
    format!(
        "mo-cuda-beta-spsa split_count={} beta_entries={} baseline_margin={:.6} \
         frontier_margin={} frontier_admitted={} final_margin={} fixed_critical_row={} \
         updates_attempted={} updates_completed={} selected_update={} final_admitted={} \
         scalar_selected={} elapsed={:.6}s",
        summary.split_count,
        summary.beta_entries,
        summary.baseline_margin,
        frontier_value(summary.frontier),
        summary.frontier_admitted,
        summary
            .final_margin
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "none".to_string()),
        option_value(summary.fixed_critical_row),
        summary.updates_attempted,
        summary.updates_completed,
        option_value(summary.selected_update),
        summary.final_admitted,
        summary.scalar_selected,
        elapsed.as_secs_f64(),
    )
}

fn emit_spsa_phase_summary(summary: SpsaPhaseSummary, started: Option<Instant>) {
    let Some(started) = started else {
        return;
    };
    crate::phase_telemetry::phase_marker(&spsa_phase_summary_line(summary, started.elapsed()));
}

fn probe_candidate_improves(candidate: f32, best: f32) -> bool {
    candidate.is_finite() && candidate > best
}

fn completed_lower(bounds: &[(f32, f32)]) -> Vec<f32> {
    bounds
        .iter()
        .map(|&(lower, _)| {
            if lower.is_finite() {
                lower
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}

fn fold_completed_lower(best: &mut [f32], bounds: &[(f32, f32)]) {
    if best.len() != bounds.len() {
        return;
    }
    for (slot, &(lower, _)) in best.iter_mut().zip(bounds) {
        if lower.is_finite() && lower > *slot {
            *slot = lower;
        }
    }
}

fn apply_completed_lower(bounds: &mut [(f32, f32)], best: &[f32]) {
    if bounds.len() != best.len() {
        return;
    }
    for (bound, &lower) in bounds.iter_mut().zip(best) {
        if lower.is_finite() && lower > bound.0 && lower <= bound.1 {
            bound.0 = lower;
        }
    }
}

fn initial_standard_result<T>(result: Result<T>, deadline_reached: bool) -> Result<T> {
    if deadline_reached {
        return Err(NyError::DeadlineExceeded(
            "multi-objective CUDA β-SPSA baseline completed after deadline".into(),
        ));
    }
    result
}

/// Follow-up optimization is optional. An on-time certified infeasibility is
/// authoritative at the child-processing boundary and must propagate; no late
/// result is admitted.
fn followup_standard_result_preserving_infeasible<T>(
    result: Result<T>,
    deadline_reached: bool,
) -> Result<Option<T>> {
    if deadline_reached {
        return Ok(None);
    }
    match result {
        Err(error) if error.is_infeasible_domain() => Err(error),
        Ok(value) => Ok(Some(value)),
        Err(_) => Ok(None),
    }
}

fn standard_bounds_are_publishable(bounds: &[(f32, f32)], expected_rows: usize) -> bool {
    bounds.len() == expected_rows
        && bounds.iter().all(|&(lower, upper)| {
            !lower.is_nan()
                && lower != f32::INFINITY
                && !upper.is_nan()
                && upper != f32::NEG_INFINITY
                && lower <= upper
        })
}

fn exact_cuda_wide_enabled() -> bool {
    std::env::var("NY_CUDA_WIDE").ok().as_deref() == Some("1")
}

/// Select the exact inherited alpha state consumed by the Standard pass.
///
/// Both ResNet preps in this lane must use the same state instead of silently
/// reverting their advisory CUDA probes to heuristic alpha slopes.
fn inherited_alpha_for_spsa_prep<'a>(
    context: &GraphCrownContext<'a>,
) -> Option<&'a GraphDomainAlphaState> {
    context.alpha_state
}

fn rows_streamable(rows: usize, capacity: usize) -> bool {
    if rows < 2 || !(2..=DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&capacity) {
        return false;
    }
    let chunks = rows.div_ceil(capacity);
    rows >= chunks.saturating_mul(2)
}

fn targets_are_rectangular_and_finite(targets: &MultiObjectiveTargets<'_>) -> bool {
    let rows = targets.objectives.len();
    if !(2..=512).contains(&rows)
        || targets.thresholds.len() != rows
        || targets.verified_mask.len() != rows
    {
        return false;
    }
    let Some(width) = targets.objectives.first().map(Vec::len) else {
        return false;
    };
    width > 0
        && targets.thresholds.iter().all(|value| value.is_finite())
        && targets
            .objectives
            .iter()
            .all(|row| row.len() == width && row.iter().all(|value| value.is_finite()))
}

fn beta_is_relu_only_and_covered(beta: &GraphBetaState, prep: &ResnetDomainPrep) -> bool {
    if beta.entries.is_empty() {
        return false;
    }
    beta.entries.iter().all(|entry| {
        entry.split_point() == 0.0
            && prep
                .relu_names
                .iter()
                .zip(&prep.beta_signed)
                .find(|(name, _)| name.as_str() == entry.node_name())
                .is_some_and(|(_, values)| entry.neuron_idx() < values.len())
    })
}

fn beta_table(beta: &GraphBetaState, prep: &ResnetDomainPrep) -> Option<Vec<Vec<f32>>> {
    if !beta_is_relu_only_and_covered(beta, prep) {
        return None;
    }
    let mut table = prep
        .beta_signed
        .iter()
        .map(|values| vec![0.0; values.len()])
        .collect::<Vec<_>>();
    for entry in &beta.entries {
        let relu = prep
            .relu_names
            .iter()
            .position(|name| name == entry.node_name())?;
        let signed = entry.signed_value();
        if !signed.is_finite() {
            return None;
        }
        let slot = &mut table[relu][entry.neuron_idx()];
        let accumulated = *slot + signed;
        if !accumulated.is_finite() {
            return None;
        }
        *slot = accumulated;
    }
    Some(table)
}

fn normalized_margin(lower: f32, threshold: f32) -> f32 {
    let margin = lower - threshold;
    if !margin.is_finite() {
        f32::NEG_INFINITY
    } else {
        margin
    }
}

/// Admit advisory SPSA only when this completed Standard child is on the
/// deterministic queued-domain proof frontier.
///
/// No queued frontier admits the first eligible child. Equality admits to
/// avoid float-dependent tie asymmetry; a strictly better (larger) margin is
/// non-binding and returns the baseline unchanged. Any non-finite input fails
/// closed and cannot authorize extra work.
fn frontier_admits_spsa(baseline_margin: f32, frontier: MoCudaBetaSpsaFrontier) -> bool {
    baseline_margin.is_finite()
        && match frontier {
            MoCudaBetaSpsaFrontier::Empty => true,
            MoCudaBetaSpsaFrontier::Finite(margin) => {
                margin.is_finite() && baseline_margin <= margin
            }
            MoCudaBetaSpsaFrontier::Invalid => false,
        }
}

#[derive(Debug, PartialEq)]
enum PostBaselineSpsa<T> {
    /// The completed Standard child is not on the proof frontier.
    OffFrontier,
    /// Optional work was authorized but refused or no authority remains.
    BaselineOnly,
    /// Optional preflight completed and SPSA may proceed.
    Ready(T),
}

/// Run expensive SPSA preflight only after one completed Standard baseline has
/// passed both the deterministic frontier gate and a fresh deadline check.
///
/// `None` from `prepare` is an explicit baseline-only outcome. In particular,
/// it must not fall through to the legacy optimizer after consuming a Standard
/// pass: that would silently double proof work under the finite deadline.
fn prepare_post_baseline_spsa<T>(
    baseline_margin: f32,
    frontier: MoCudaBetaSpsaFrontier,
    deadline_reached: bool,
    prepare: impl FnOnce() -> Option<T>,
) -> PostBaselineSpsa<T> {
    if !frontier_admits_spsa(baseline_margin, frontier) {
        return PostBaselineSpsa::OffFrontier;
    }
    if deadline_reached {
        return PostBaselineSpsa::BaselineOnly;
    }
    prepare().map_or(PostBaselineSpsa::BaselineOnly, PostBaselineSpsa::Ready)
}

fn objective_margin(bounds: &[(f32, f32)], targets: &MultiObjectiveTargets<'_>) -> Option<f32> {
    (bounds.len() == targets.objectives.len()).then(|| {
        bounds
            .iter()
            .zip(targets.thresholds)
            .zip(targets.verified_mask)
            .filter(|(_, verified)| !**verified)
            .map(|((bound, threshold), _)| normalized_margin(bound.0, *threshold))
            .fold(f32::INFINITY, f32::min)
    })
}

fn critical_row(bounds: &[(f32, f32)], targets: &MultiObjectiveTargets<'_>) -> Option<usize> {
    if bounds.len() != targets.objectives.len() {
        return None;
    }
    bounds
        .iter()
        .zip(targets.thresholds)
        .zip(targets.verified_mask)
        .enumerate()
        .filter(|(_, (_, verified))| !**verified)
        .map(|(row, ((bound, threshold), _))| (row, normalized_margin(bound.0, *threshold)))
        .min_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        })
        .map(|(row, _)| row)
}

fn spsa_direction(update: usize, coordinate: usize) -> f32 {
    let mut value = (coordinate as u64)
        .wrapping_add(1)
        .wrapping_add((update as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    if value & 1 == 0 {
        -1.0
    } else {
        1.0
    }
}

fn projected_pair(
    beta: &GraphBetaState,
    update: usize,
) -> (GraphBetaState, GraphBetaState, Vec<f32>) {
    let mut plus = beta.clone();
    let mut minus = beta.clone();
    let mut spans = Vec::with_capacity(beta.entries.len());
    for coordinate in 0..beta.entries.len() {
        let value = beta.entries[coordinate].value();
        let delta = SPSA_EPSILON * spsa_direction(update, coordinate);
        let plus_value = (value + delta).max(0.0);
        let minus_value = (value - delta).max(0.0);
        plus.entries[coordinate].set_value(plus_value);
        minus.entries[coordinate].set_value(minus_value);
        spans.push(plus_value - minus_value);
    }
    (plus, minus, spans)
}

fn run_advisory_probe_loop(
    mut current: GraphBetaState,
    baseline_score: f32,
    updates: usize,
    adaptive_config: &AdaptiveOptConfig,
    tolerance: f32,
    deadline: Instant,
    mut probe: impl FnMut(&GraphBetaState) -> Result<f32>,
) -> ProbeLoopOutcome {
    let mut best = ProbeBestSnapshot {
        score: baseline_score,
        update: None,
        beta: current.clone(),
    };
    let mut attempted = 0usize;
    let mut completed = 0usize;
    for update in 0..updates.min(MAX_SPSA_UPDATES) {
        if Instant::now() >= deadline {
            break;
        }
        attempted += 1;
        let (plus, minus, spans) = projected_pair(&current, update);
        if Instant::now() >= deadline {
            break;
        }
        let plus_value = match probe(&plus) {
            Ok(value) => value,
            Err(_) => break,
        };
        let minus_value = match probe(&minus) {
            Ok(value) => value,
            Err(_) => break,
        };
        let difference = plus_value - minus_value;
        if !difference.is_finite() {
            break;
        }
        current.zero_grad();
        for (entry, span) in current.entries.iter_mut().zip(spans) {
            let gradient = if span == 0.0 { 0.0 } else { difference / span };
            entry.grad = if gradient.is_finite() { gradient } else { 0.0 };
        }
        let max_grad = current.gradient_step_adam(adaptive_config, update + 1);
        if Instant::now() >= deadline {
            break;
        }
        let post_update = match probe(&current) {
            Ok(value) => value,
            Err(_) => break,
        };
        completed += 1;
        best.consider_post_update(post_update, update + 1, &current);
        if max_grad < tolerance {
            break;
        }
    }
    ProbeLoopOutcome {
        best,
        attempted,
        completed,
    }
}

fn run_optional_final<T>(
    selected_update: Option<usize>,
    authority_remaining: bool,
    certify: impl FnOnce() -> T,
) -> Option<T> {
    (selected_update.is_some() && authority_remaining).then(certify)
}

fn probe_critical_row(
    gpu: &dyn GpuCrownBackward,
    prep: &ResnetDomainPrep,
    objective: &[f32],
    beta: &GraphBetaState,
    deadline: Instant,
) -> Result<f32> {
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(
            "multi-objective CUDA β-SPSA probe started after deadline".into(),
        ));
    }
    if gpu.deadline_bounded_resnet_sound_beta_max_rows() < 2 {
        return Err(NyError::UnsupportedOp(
            "multi-objective CUDA β-SPSA requires two-row beta capability".into(),
        ));
    }
    let beta_signed = beta_table(beta, prep).ok_or_else(|| {
        NyError::InvalidSpec("CUDA β-SPSA beta table does not match ResNet fold".into())
    })?;
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(
            "multi-objective CUDA β-SPSA probe beta-table preparation exceeded deadline".into(),
        ));
    }
    let mut rows = Vec::with_capacity(objective.len().saturating_mul(2));
    rows.extend_from_slice(objective);
    rows.extend_from_slice(objective);
    let seed = GpuCrownSeed {
        lower_a: rows.clone().into(),
        upper_a: rows.into(),
        lower_b: vec![0.0; 2].into(),
        upper_b: vec![0.0; 2].into(),
        num_specs: 2,
        current_dim: objective.len(),
    };
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(
            "multi-objective CUDA β-SPSA probe seed preparation exceeded deadline".into(),
        ));
    }
    let result = gpu.crown_backward_gpu_resnet_sound_beta_bounded_rows_with_deadline(
        &prep.segments,
        &seed,
        &prep.in_lo,
        &prep.in_hi,
        &beta_signed,
        &prep.frontier_abs,
        &prep.node_abs,
        deadline,
    )?;
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(
            "multi-objective CUDA β-SPSA probe completed after deadline".into(),
        ));
    }
    if result.lower_bounds.len() != 2
        || result.upper_bounds.len() != 2
        || result
            .lower_bounds
            .iter()
            .zip(&result.upper_bounds)
            .any(|(&lower, &upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
    {
        return Err(NyError::InvalidSpec(
            "multi-objective CUDA β-SPSA probe returned malformed bounds".into(),
        ));
    }
    Ok(result.lower_bounds[0].min(result.lower_bounds[1]))
}

impl BetaCrownVerifier {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_optimize_multi_objective_cuda_beta_spsa(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &mut GraphBetaState,
        targets: &MultiObjectiveTargets<'_>,
        conjunctive: bool,
        seed_caches: &[Option<&CachedLinearBounds>],
        frontier: MoCudaBetaSpsaFrontier,
    ) -> Option<Result<MultiObjectiveWarmStartResult>> {
        let deadline = self.effective_graph_bab_deadline()?;
        // Cheap, side-effect-free refusals retain the exact legacy
        // algorithm/state path. ResNet construction and backend observation
        // are intentionally deferred until after Standard frontier admission.
        if conjunctive
            || !exact_cuda_wide_enabled()
            || !crate::network::resnet_beta_gpu_enabled()
            || !context.history.genbab_constraints.is_empty()
            || context.history.has_norm_inv_rms_constraints()
            || context.base_bounds.is_none()
            || beta_state
                .entries
                .iter()
                .any(|entry| entry.split_point() != 0.0)
            || !targets_are_rectangular_and_finite(targets)
            || Instant::now() >= deadline
        {
            return None;
        }

        let telemetry_started =
            crate::phase_telemetry::phase_telemetry_enabled().then(Instant::now);
        Some((|| {
            let current = beta_state.clone();
            let initial = self.propagate_multi_objective_with_beta_and_cache(
                graph,
                input,
                context,
                &current,
                targets,
                seed_caches,
                false,
            );
            let (bounds, node_bounds, _) =
                initial_standard_result(initial, Instant::now() >= deadline)?;
            if !standard_bounds_are_publishable(&bounds, targets.objectives.len()) {
                return Err(NyError::InvalidSpec(
                    "CUDA β-SPSA baseline Standard result is malformed".into(),
                ));
            }
            let margin = objective_margin(&bounds, targets).ok_or_else(|| {
                NyError::InvalidSpec("CUDA β-SPSA Standard result row mismatch".into())
            })?;
            let baseline_margin = margin;
            let beta_entries = beta_state.entries.len();

            // The Standard baseline and proof-frontier admission deliberately
            // precede every ResNet prep and backend lookup. The backend accessor
            // observes only a preinitialized engine, so neither a cold CUDA
            // factory nor an off-frontier child can consume optional authority.
            let prepared = prepare_post_baseline_spsa(
                baseline_margin,
                frontier,
                Instant::now() >= deadline,
                || {
                    let fixed_row = critical_row(&bounds, targets)?;
                    let gpu = crate::sound_gpu_gate::preinitialized_sound_gpu_crown_for_wide()?;
                    let capacity = gpu.deadline_bounded_resnet_sound_beta_max_rows();
                    if !rows_streamable(targets.objectives.len(), capacity)
                        || Instant::now() >= deadline
                    {
                        return None;
                    }
                    let prep = prep_resnet_domain_ext(
                        graph,
                        &graph.output_node,
                        &node_bounds,
                        input,
                        Some(&current),
                        inherited_alpha_for_spsa_prep(context),
                        false,
                        false,
                        false,
                    );
                    if Instant::now() >= deadline {
                        return None;
                    }
                    let prep = prep?;
                    beta_is_relu_only_and_covered(&current, &prep).then_some((fixed_row, gpu, prep))
                },
            );
            let (fixed_row, gpu, fixed_prep) = match prepared {
                PostBaselineSpsa::OffFrontier => {
                    // This is intentionally the exact baseline-only return
                    // shape: no prep, backend access, advisory probe, beta
                    // mutation, rowwise fold, or final Standard pass has run.
                    emit_spsa_phase_summary(
                        SpsaPhaseSummary {
                            split_count: context.history.split_count,
                            beta_entries,
                            baseline_margin,
                            frontier,
                            frontier_admitted: false,
                            final_margin: None,
                            fixed_critical_row: None,
                            updates_attempted: 0,
                            updates_completed: 0,
                            selected_update: None,
                            final_admitted: false,
                            scalar_selected: false,
                        },
                        telemetry_started,
                    );
                    tracing::info!(
                        objectives = targets.objectives.len(),
                        baseline_margin,
                        ?frontier,
                        "Multi-objective β optimization: CUDA SPSA skipped off proof frontier"
                    );
                    return Ok((bounds, node_bounds, vec![None; targets.objectives.len()]));
                }
                PostBaselineSpsa::BaselineOnly => {
                    // Once Standard has completed, a late, cold, unsupported,
                    // or structurally ineligible optional preflight returns that
                    // one certificate. It never falls through and double-runs
                    // Standard inside the legacy optimizer.
                    emit_spsa_phase_summary(
                        SpsaPhaseSummary {
                            split_count: context.history.split_count,
                            beta_entries,
                            baseline_margin,
                            frontier,
                            frontier_admitted: true,
                            final_margin: None,
                            fixed_critical_row: None,
                            updates_attempted: 0,
                            updates_completed: 0,
                            selected_update: None,
                            final_admitted: false,
                            scalar_selected: false,
                        },
                        telemetry_started,
                    );
                    tracing::info!(
                        objectives = targets.objectives.len(),
                        baseline_margin,
                        "Multi-objective β optimization: CUDA SPSA optional preflight refused; \
                         returning completed Standard baseline"
                    );
                    return Ok((bounds, node_bounds, vec![None; targets.objectives.len()]));
                }
                PostBaselineSpsa::Ready(prepared) => prepared,
            };
            let mut best_lower = completed_lower(&bounds);
            let mut best = CompletedStandard {
                margin,
                bounds,
                node_bounds,
                beta: current.clone(),
            };

            // Freeze both the certified baseline relaxation and its worst
            // unverified row. Every SPSA call below is advisory and uses this
            // one sound decomposition; only the optional final Standard pass
            // can publish a changed beta.
            let baseline_probe_score = best.bounds[fixed_row].0;
            let probe_loop = if Instant::now() < deadline {
                run_advisory_probe_loop(
                    current,
                    baseline_probe_score,
                    self.config.beta_iterations,
                    &self.config.adaptive_config,
                    self.config.beta_tolerance,
                    deadline,
                    |candidate| {
                        probe_critical_row(
                            gpu,
                            &fixed_prep,
                            &targets.objectives[fixed_row],
                            candidate,
                            deadline,
                        )
                    },
                )
            } else {
                ProbeLoopOutcome {
                    best: ProbeBestSnapshot {
                        score: baseline_probe_score,
                        update: None,
                        beta: current,
                    },
                    attempted: 0,
                    completed: 0,
                }
            };
            let updates_attempted = probe_loop.attempted;
            let updates_completed = probe_loop.completed;
            let probe_best = probe_loop.best;
            let selected_update = probe_best.update;
            let mut final_margin = None;
            let mut final_admitted = false;
            let mut scalar_selected = false;

            // A baseline selection is already certified, so it needs no second
            // Standard pass. Otherwise certify exactly the probe-best update,
            // even when a later advisory probe failed, while authority remains.
            let final_evaluation =
                run_optional_final(selected_update, Instant::now() < deadline, || {
                    self.propagate_multi_objective_with_beta_and_cache(
                        graph,
                        input,
                        context,
                        &probe_best.beta,
                        targets,
                        seed_caches,
                        false,
                    )
                });
            if let Some(evaluated) = final_evaluation {
                match followup_standard_result_preserving_infeasible(
                    evaluated,
                    Instant::now() >= deadline,
                ) {
                    Err(error) => {
                        // `followup_standard_result_preserving_infeasible`
                        // performed the literal post-call authority check
                        // before reaching this verdict-neutral stderr write.
                        emit_spsa_phase_summary(
                            SpsaPhaseSummary {
                                split_count: context.history.split_count,
                                beta_entries,
                                baseline_margin,
                                frontier,
                                frontier_admitted: true,
                                final_margin: None,
                                fixed_critical_row: Some(fixed_row),
                                updates_attempted,
                                updates_completed,
                                selected_update,
                                final_admitted: false,
                                scalar_selected: false,
                            },
                            telemetry_started,
                        );
                        return Err(error);
                    }
                    Ok(Some((bounds, node_bounds, _)))
                        if standard_bounds_are_publishable(&bounds, targets.objectives.len()) =>
                    {
                        let margin = objective_margin(&bounds, targets).ok_or_else(|| {
                            NyError::InvalidSpec(
                                "CUDA β-SPSA final Standard result row mismatch".into(),
                            )
                        })?;
                        final_margin = Some(margin);
                        final_admitted = true;
                        fold_completed_lower(&mut best_lower, &bounds);
                        if margin > best.margin {
                            best = CompletedStandard {
                                margin,
                                bounds,
                                node_bounds,
                                beta: probe_best.beta,
                            };
                            scalar_selected = true;
                        }
                    }
                    Ok(Some(_)) | Ok(None) => {}
                }
            }

            *beta_state = best.beta;
            // The warm-start beta remains the scalar-margin winner. Each merged
            // lower row is an independent completed Standard certificate for
            // the same child and need not be reproduced by that one snapshot.
            apply_completed_lower(&mut best.bounds, &best_lower);
            emit_spsa_phase_summary(
                SpsaPhaseSummary {
                    split_count: context.history.split_count,
                    beta_entries,
                    baseline_margin,
                    frontier,
                    frontier_admitted: true,
                    final_margin,
                    fixed_critical_row: Some(fixed_row),
                    updates_attempted,
                    updates_completed,
                    selected_update,
                    final_admitted,
                    scalar_selected,
                },
                telemetry_started,
            );
            tracing::info!(
                objectives = targets.objectives.len(),
                updates_attempted,
                updates_completed,
                "Multi-objective β optimization: completed bounded CUDA SPSA lane"
            );
            Ok((
                best.bounds,
                best.node_bounds,
                vec![None; targets.objectives.len()],
            ))
        })())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::time::{Duration, Instant};

    use ny_core::{GpuCrownBackward, GpuCrownResult, GpuCrownSeed, NyError, Result};

    use super::{
        apply_completed_lower, beta_table, completed_lower, fold_completed_lower,
        followup_standard_result_preserving_infeasible, frontier_admits_spsa,
        inherited_alpha_for_spsa_prep, initial_standard_result, normalized_margin,
        prepare_post_baseline_spsa, probe_critical_row, projected_pair, rows_streamable,
        run_advisory_probe_loop, run_optional_final, spsa_direction, spsa_phase_summary_line,
        standard_bounds_are_publishable, MoCudaBetaSpsaFrontier, PostBaselineSpsa,
        ProbeBestSnapshot, ResnetDomainPrep, SpsaPhaseSummary, SPSA_EPSILON,
    };
    use crate::beta_crown::branching::GraphSplitHistory;
    use crate::beta_crown::config::AdaptiveOptConfig;
    use crate::beta_crown::domain::GraphCrownContext;
    use crate::beta_crown::state::{
        AlphaNeuronState, GraphBetaEntry, GraphBetaState, GraphDomainAlphaState,
    };

    struct ProbeGpu {
        malformed: bool,
        fail: bool,
    }

    impl GpuCrownBackward for ProbeGpu {
        fn crown_backward_gpu(
            &self,
            _layers: &[ny_core::GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            Err(NyError::UnsupportedOp("ordinary test route".into()))
        }

        fn deadline_bounded_resnet_sound_beta_max_rows(&self) -> usize {
            2
        }

        fn crown_backward_gpu_resnet_sound_beta_bounded_rows_with_deadline(
            &self,
            _segments: &[ny_core::GpuResnetSegment],
            seed: &GpuCrownSeed,
            _input_lower: &[f32],
            _input_upper: &[f32],
            beta_signed: &[Vec<f32>],
            _frontier_abs: &[Vec<f32>],
            _node_abs: &[Vec<f32>],
            _deadline: Instant,
        ) -> Result<GpuCrownResult> {
            assert_eq!(seed.num_specs, 2);
            assert_eq!(
                &seed.lower_a[..seed.current_dim],
                &seed.lower_a[seed.current_dim..]
            );
            assert_eq!(beta_signed, &[vec![0.25, 0.0]]);
            if self.fail {
                return Err(NyError::InternalError("injected probe refusal".into()));
            }
            if self.malformed {
                return Ok(GpuCrownResult {
                    lower_bounds: vec![f32::NAN, 1.0],
                    upper_bounds: vec![2.0, 2.0],
                });
            }
            Ok(GpuCrownResult {
                lower_bounds: vec![1.25, 1.5],
                upper_bounds: vec![2.0, 2.0],
            })
        }
    }

    fn probe_prep() -> ResnetDomainPrep {
        ResnetDomainPrep {
            segments: Vec::new(),
            relu_names: vec!["relu".into()],
            frontier_abs: Vec::new(),
            node_abs: Vec::new(),
            beta_signed: vec![vec![0.0, 0.0]],
            alpha_bridge: None,
            in_lo: Vec::new(),
            in_hi: Vec::new(),
            stop_node: None,
        }
    }

    fn probe_beta() -> GraphBetaState {
        GraphBetaState::from_entries(vec![
            GraphBetaEntry::new("relu".into(), 0, 0.0, 0.25, 1.0).unwrap()
        ])
    }

    #[test]
    fn prep_policy_preserves_inherited_alpha_identity_and_absence() {
        let history = GraphSplitHistory::new();
        let mut alpha = GraphDomainAlphaState::empty();
        alpha.insert("relu".into(), 0, AlphaNeuronState::new(0.375));
        let inherited = GraphCrownContext::for_history(&history).with_alpha(&alpha);
        let selected =
            inherited_alpha_for_spsa_prep(&inherited).expect("inherited alpha must be selected");
        assert!(std::ptr::eq(selected, &raw const alpha));
        assert_eq!(selected.alpha("relu", 0).to_bits(), 0.375_f32.to_bits());

        let absent = GraphCrownContext::for_history(&history);
        assert!(inherited_alpha_for_spsa_prep(&absent).is_none());
    }

    #[test]
    fn bounded_rows_must_partition_without_singletons() {
        assert!(rows_streamable(2, 8));
        assert!(rows_streamable(9, 8));
        assert!(!rows_streamable(1, 8));
        assert!(!rows_streamable(3, 2));
        assert!(!rows_streamable(9, 9));
    }

    #[test]
    fn normalized_margin_maps_every_nonfinite_result_to_negative_infinity() {
        assert_eq!(normalized_margin(3.0, 1.0), 2.0);
        for margin in [
            normalized_margin(f32::NAN, 0.0),
            normalized_margin(f32::INFINITY, 0.0),
            normalized_margin(f32::NEG_INFINITY, 0.0),
            normalized_margin(f32::MAX, -f32::MAX),
        ] {
            assert_eq!(margin, f32::NEG_INFINITY);
        }
    }

    #[test]
    fn frontier_admission_accepts_none_equal_and_below_but_rejects_above() {
        assert!(frontier_admits_spsa(-2.0, MoCudaBetaSpsaFrontier::Empty));
        assert!(frontier_admits_spsa(
            -2.0,
            MoCudaBetaSpsaFrontier::Finite(-2.0)
        ));
        assert!(frontier_admits_spsa(
            -3.0,
            MoCudaBetaSpsaFrontier::Finite(-2.0)
        ));
        assert!(!frontier_admits_spsa(
            -1.0,
            MoCudaBetaSpsaFrontier::Finite(-2.0)
        ));
    }

    #[test]
    fn frontier_admission_rejects_every_nonfinite_authority() {
        for baseline in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            assert!(!frontier_admits_spsa(
                baseline,
                MoCudaBetaSpsaFrontier::Empty
            ));
            assert!(!frontier_admits_spsa(
                baseline,
                MoCudaBetaSpsaFrontier::Finite(-2.0)
            ));
        }
        for frontier in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            assert!(!frontier_admits_spsa(
                -2.0,
                MoCudaBetaSpsaFrontier::Finite(frontier)
            ));
        }
        assert!(!frontier_admits_spsa(-2.0, MoCudaBetaSpsaFrontier::Invalid));
    }

    #[test]
    fn post_baseline_gate_skips_all_optional_work_off_frontier_or_expired() {
        for (frontier, deadline_reached) in [
            (MoCudaBetaSpsaFrontier::Finite(-2.0), false),
            (MoCudaBetaSpsaFrontier::Empty, true),
        ] {
            let standard_calls = Cell::new(0usize);
            let prep_calls = Cell::new(0usize);
            let backend_calls = Cell::new(0usize);

            // Models the one production Standard call immediately before the
            // helper. Everything inside `prepare` is post-frontier optional work.
            standard_calls.set(standard_calls.get() + 1);
            let decision = prepare_post_baseline_spsa(-1.0, frontier, deadline_reached, || {
                prep_calls.set(prep_calls.get() + 1);
                backend_calls.set(backend_calls.get() + 1);
                Some(7usize)
            });

            assert!(matches!(
                decision,
                PostBaselineSpsa::OffFrontier | PostBaselineSpsa::BaselineOnly
            ));
            assert_eq!(standard_calls.get(), 1);
            assert_eq!(prep_calls.get(), 0);
            assert_eq!(backend_calls.get(), 0);
        }
    }

    #[test]
    fn admitted_preflight_refusal_is_baseline_only_without_a_second_standard() {
        let standard_calls = Cell::new(0usize);
        let backend_calls = Cell::new(0usize);
        standard_calls.set(standard_calls.get() + 1);

        let decision =
            prepare_post_baseline_spsa(-2.0, MoCudaBetaSpsaFrontier::Empty, false, || {
                backend_calls.set(backend_calls.get() + 1);
                None::<usize>
            });

        assert_eq!(decision, PostBaselineSpsa::BaselineOnly);
        assert_eq!(backend_calls.get(), 1);
        assert_eq!(
            standard_calls.get(),
            1,
            "a cold or structurally ineligible optional preflight must publish \
             its completed baseline instead of falling through to legacy Standard"
        );
    }

    #[test]
    fn projected_pair_is_deterministic_and_nonnegative() {
        let beta = GraphBetaState::from_entries(vec![
            GraphBetaEntry::new("relu".into(), 0, 0.0, 0.0, 1.0).unwrap(),
            GraphBetaEntry::new("relu".into(), 1, 0.0, 0.5, -1.0).unwrap(),
        ]);
        let (plus_a, minus_a, spans_a) = projected_pair(&beta, 1);
        let (plus_b, minus_b, spans_b) = projected_pair(&beta, 1);
        assert_eq!(spans_a, spans_b);
        for (left, right) in plus_a.entries.iter().zip(&plus_b.entries) {
            assert_eq!(left.value(), right.value());
            assert!(left.value() >= 0.0);
        }
        for (left, right) in minus_a.entries.iter().zip(&minus_b.entries) {
            assert_eq!(left.value(), right.value());
            assert!(left.value() >= 0.0);
        }
        assert!(spans_a.iter().all(|span| span.abs() <= 2.0 * SPSA_EPSILON));
        assert!([-1.0, 1.0].contains(&spsa_direction(1, 1)));
    }

    #[test]
    fn post_update_selection_retains_baseline_until_a_strict_probe_improvement() {
        let baseline = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "relu".into(),
            0,
            0.0,
            0.0,
            1.0,
        )
        .unwrap()]);
        let mut selection = ProbeBestSnapshot {
            score: -5.0,
            update: None,
            beta: baseline,
        };
        let update_one = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "relu".into(),
            0,
            0.0,
            0.25,
            1.0,
        )
        .unwrap()]);
        assert!(selection.consider_post_update(-4.0, 1, &update_one));
        assert_eq!(selection.update, Some(1));
        assert_eq!(selection.beta.entries[0].value(), 0.25);

        let worse = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "relu".into(),
            0,
            0.0,
            0.75,
            1.0,
        )
        .unwrap()]);
        assert!(!selection.consider_post_update(-6.0, 2, &worse));
        assert!(!selection.consider_post_update(f32::NAN, 2, &worse));
        assert_eq!(selection.update, Some(1));
        assert_eq!(selection.beta.entries[0].value(), 0.25);

        assert!(selection.consider_post_update(-3.0, 3, &worse));
        assert_eq!(selection.update, Some(3));
        assert_eq!(selection.beta.entries[0].value(), 0.75);
    }

    #[test]
    fn advisory_orchestration_runs_three_probes_per_update_and_retains_last_best_on_failure() {
        let initial = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "relu".into(),
            0,
            0.0,
            0.0,
            1.0,
        )
        .unwrap()]);
        let responses = [2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0, 3.0];
        let mut calls = 0usize;
        let outcome = run_advisory_probe_loop(
            initial.clone(),
            0.0,
            99,
            &AdaptiveOptConfig::default(),
            -1.0,
            Instant::now() + Duration::from_secs(5),
            |_| {
                let value = responses[calls];
                calls += 1;
                Ok(value)
            },
        );
        assert_eq!(calls, 9, "three updates require plus/minus/post probes");
        assert_eq!(outcome.attempted, 3);
        assert_eq!(outcome.completed, 3);
        assert_eq!(outcome.best.update, Some(3));

        let mut failed_calls = 0usize;
        let failed = run_advisory_probe_loop(
            initial.clone(),
            0.0,
            3,
            &AdaptiveOptConfig::default(),
            -1.0,
            Instant::now() + Duration::from_secs(5),
            |_| {
                failed_calls += 1;
                if failed_calls == 5 {
                    Err(NyError::InternalError("later probe failure".into()))
                } else {
                    Ok(match failed_calls {
                        1 | 4 => 2.0,
                        2 => 0.0,
                        3 => 1.0,
                        _ => unreachable!(),
                    })
                }
            },
        );
        assert_eq!(failed_calls, 5);
        assert_eq!(failed.attempted, 2);
        assert_eq!(failed.completed, 1);
        assert_eq!(failed.best.update, Some(1));

        let baseline_only = run_advisory_probe_loop(
            initial,
            0.0,
            3,
            &AdaptiveOptConfig::default(),
            -1.0,
            Instant::now() + Duration::from_secs(5),
            |_| Err(NyError::InternalError("first probe failure".into())),
        );
        assert_eq!(baseline_only.attempted, 1);
        assert_eq!(baseline_only.completed, 0);
        assert_eq!(baseline_only.best.update, None);
        assert_eq!(baseline_only.best.beta.entries[0].value(), 0.0);
    }

    #[test]
    fn optional_final_certification_runs_at_most_once_and_skips_baseline_selection() {
        let calls = Cell::new(0usize);
        let result = run_optional_final(Some(2), true, || {
            calls.set(calls.get() + 1);
            7usize
        });
        assert_eq!(result, Some(7));
        assert_eq!(calls.get(), 1);

        assert_eq!(
            run_optional_final(None, true, || {
                calls.set(calls.get() + 1);
                8usize
            }),
            None
        );
        assert_eq!(
            run_optional_final(Some(2), false, || {
                calls.set(calls.get() + 1);
                9usize
            }),
            None
        );
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn phase_summary_has_stable_required_fields() {
        let line = spsa_phase_summary_line(
            SpsaPhaseSummary {
                split_count: 4,
                beta_entries: 3,
                baseline_margin: -2.5,
                frontier: MoCudaBetaSpsaFrontier::Finite(-2.0),
                frontier_admitted: true,
                final_margin: Some(-1.25),
                fixed_critical_row: Some(7),
                updates_attempted: 3,
                updates_completed: 2,
                selected_update: Some(1),
                final_admitted: true,
                scalar_selected: false,
            },
            Duration::from_millis(1250),
        );
        assert_eq!(
            line,
            "mo-cuda-beta-spsa split_count=4 beta_entries=3 baseline_margin=-2.500000 \
             frontier_margin=-2.000000 frontier_admitted=true final_margin=-1.250000 \
             fixed_critical_row=7 updates_attempted=3 updates_completed=2 \
             selected_update=1 final_admitted=true scalar_selected=false elapsed=1.250000s"
        );
    }

    #[test]
    fn beta_table_maps_active_inactive_and_accumulates_duplicate_neurons() {
        let beta = GraphBetaState::from_entries(vec![
            GraphBetaEntry::new("relu".into(), 0, 0.0, 0.75, 1.0).unwrap(),
            GraphBetaEntry::new("relu".into(), 1, 0.0, 0.4, -1.0).unwrap(),
            GraphBetaEntry::new("relu".into(), 0, 0.0, 0.25, -1.0).unwrap(),
        ]);
        let table = beta_table(&beta, &probe_prep()).expect("covered finite beta table");
        assert_eq!(table, vec![vec![0.5, -0.4]]);
        assert_eq!(table[0][0], beta.signed_beta("relu", 0).unwrap());
        assert_eq!(table[0][1], beta.signed_beta("relu", 1).unwrap());

        let overflowing = GraphBetaState::from_entries(vec![
            GraphBetaEntry::new("relu".into(), 0, 0.0, f32::MAX, 1.0).unwrap(),
            GraphBetaEntry::new("relu".into(), 0, 0.0, f32::MAX, 1.0).unwrap(),
        ]);
        assert!(
            beta_table(&overflowing, &probe_prep()).is_none(),
            "non-finite duplicate accumulation must refuse backend admission"
        );
    }

    #[test]
    fn probe_uses_two_duplicate_rows_and_rejects_malformed_or_failed_results() {
        let deadline = Instant::now() + Duration::from_secs(5);
        let value = probe_critical_row(
            &ProbeGpu {
                malformed: false,
                fail: false,
            },
            &probe_prep(),
            &[1.0, -1.0],
            &probe_beta(),
            deadline,
        )
        .expect("valid bounded probe");
        assert_eq!(value, 1.25);

        for gpu in [
            ProbeGpu {
                malformed: true,
                fail: false,
            },
            ProbeGpu {
                malformed: false,
                fail: true,
            },
        ] {
            assert!(
                probe_critical_row(&gpu, &probe_prep(), &[1.0, -1.0], &probe_beta(), deadline,)
                    .is_err()
            );
        }
    }

    #[test]
    fn standard_publication_requires_authority_and_preserves_on_time_infeasible() {
        assert_eq!(initial_standard_result(Ok(7usize), false).unwrap(), 7);
        assert!(initial_standard_result(Ok(7usize), true).is_err());
        assert!(initial_standard_result::<usize>(
            Err(NyError::InternalError("initial".into())),
            false
        )
        .is_err());
        let late_infeasible = initial_standard_result::<usize>(
            Err(NyError::InfeasibleDomain("late empty child".into())),
            true,
        )
        .unwrap_err();
        assert!(late_infeasible.is_deadline_exceeded());

        assert_eq!(
            followup_standard_result_preserving_infeasible(Ok(9usize), false).unwrap(),
            Some(9)
        );
        assert_eq!(
            followup_standard_result_preserving_infeasible(Ok(9usize), true).unwrap(),
            None
        );
        assert_eq!(
            followup_standard_result_preserving_infeasible::<usize>(
                Err(NyError::InternalError("follow-up".into())),
                false
            )
            .unwrap(),
            None
        );
        let on_time_infeasible = followup_standard_result_preserving_infeasible::<usize>(
            Err(NyError::InfeasibleDomain("certified empty child".into())),
            false,
        )
        .unwrap_err();
        assert!(on_time_infeasible.is_infeasible_domain());
        assert_eq!(
            followup_standard_result_preserving_infeasible::<usize>(
                Err(NyError::InfeasibleDomain("late empty child".into())),
                true,
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn standard_publication_rejects_wrong_rows_nonfinite_and_inverted_intervals() {
        assert!(standard_bounds_are_publishable(
            &[(0.0, 1.0), (2.0, 2.0)],
            2
        ));
        assert!(standard_bounds_are_publishable(
            &[(f32::NEG_INFINITY, f32::INFINITY), (0.0, f32::INFINITY)],
            2
        ));
        assert!(!standard_bounds_are_publishable(&[(0.0, 1.0)], 2));
        for malformed in [
            vec![(f32::NAN, 1.0), (0.0, 1.0)],
            vec![(f32::INFINITY, f32::INFINITY), (0.0, 1.0)],
            vec![(f32::NEG_INFINITY, f32::NEG_INFINITY), (0.0, 1.0)],
            vec![(2.0, 1.0), (0.0, 1.0)],
        ] {
            assert!(!standard_bounds_are_publishable(&malformed, 2));
        }
    }

    #[test]
    fn publication_merges_only_finite_lowers_from_completed_standard_passes() {
        let mut lower = completed_lower(&[(1.0, 10.0), (5.0, 10.0), (f32::NAN, 10.0)]);
        fold_completed_lower(&mut lower, &[(3.0, 10.0), (4.0, 10.0), (6.0, 10.0)]);
        let mut scalar_winner = vec![(1.0, 10.0), (5.0, 10.0), (f32::NEG_INFINITY, 10.0)];
        apply_completed_lower(&mut scalar_winner, &lower);
        assert_eq!(scalar_winner, vec![(3.0, 10.0), (5.0, 10.0), (6.0, 10.0)]);

        fold_completed_lower(
            &mut lower,
            &[
                (f32::INFINITY, f32::INFINITY),
                (f32::NAN, 10.0),
                (f32::NEG_INFINITY, 10.0),
            ],
        );
        assert_eq!(lower, vec![3.0, 5.0, 6.0]);
    }
}
