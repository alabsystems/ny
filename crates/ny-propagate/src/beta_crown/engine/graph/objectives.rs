// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::trace;

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::domain::{GraphCrownContext, MultiObjectiveTargets, NodeBoundsView};
use crate::beta_crown::state::GraphBetaState;
use crate::bounds::{
    certified_affine_sum_f32_with_poll, nan_propagating_max, nan_propagating_min, OutwardDirection,
};
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

use crate::GraphNetwork;

use super::super::BetaCrownVerifier;
#[cfg(test)]
use super::MultiObjectiveResult;

mod cuda_beta_spsa;

type MultiObjectiveWarmStartResult = (
    Vec<(f32, f32)>,
    HashMap<String, Arc<BoundedTensor>>,
    Vec<Option<CachedLinearBounds>>,
);

/// Compute scalar objective bounds from output tensor using interval arithmetic.
///
/// Given an output `BoundedTensor` and a linear objective vector `c`, computes
/// `[lower, upper]` bounds on `c^T y` where `y` ranges over the output interval.
///
/// Uses the standard interval arithmetic rule: for coefficient `c_i >= 0`, accumulate
/// `c_i * lower_i` for the lower bound and `c_i * upper_i` for the upper bound;
/// for `c_i < 0`, swap lower and upper.
///
/// SOUNDNESS (#concretize-soundness-hardening): this result feeds
/// `domain_is_verified` at the root early-exit, where an inward round-to-nearest
/// endpoint would be an undetectable, terminal false Verified. Accumulate in f64
/// through the shared certified double-double reducer, with a per-addition
/// directed-f64 fallback if the runtime EFT self-check refuses authority.
/// Final-only widening is insufficient when cancellation follows a lost term.
///
/// ENSURES: `lower <= c^T y <= upper` for every `y` in the output box.
pub(super) fn objective_bounds(output: &BoundedTensor, objective: &[f32]) -> Result<(f32, f32)> {
    objective_bounds_with_poll(output, objective, |_| Ok(()))
}

fn objective_bounds_with_poll<P>(
    output: &BoundedTensor,
    objective: &[f32],
    mut poll: P,
) -> Result<(f32, f32)>
where
    P: FnMut(usize) -> Result<()>,
{
    if output.len() != objective.len() {
        return Err(NyError::shape_mismatch(
            vec![objective.len()],
            vec![output.len()],
        ));
    }

    // `BoundedTensor::new_unchecked` exists for trusted/internal construction,
    // so treat this verification boundary as a final proof-carrier firewall.
    // A malformed endpoint must invalidate the *whole* objective interval: if
    // only the reduction that consumes that endpoint is widened, the opposite
    // finite endpoint can still be trusted by a later comparison even though
    // the source interval was not a valid box. Properly oriented infinite
    // endpoints (`lower = -inf` or `upper = +inf`) are valid conservative
    // bounds, but impossible infinity polarity, NaN, inverted intervals, and
    // non-finite objective coefficients are not meaningful affine objectives.
    let mut malformed_input = false;
    for ((&coefficient, &lower), &upper) in objective
        .iter()
        .zip(output.lower().iter())
        .zip(output.upper().iter())
    {
        malformed_input |= !coefficient.is_finite()
            || lower.is_nan()
            || upper.is_nan()
            || lower == f32::INFINITY
            || upper == f32::NEG_INFINITY
            || lower > upper;
        poll(1)?;
    }
    if malformed_input {
        return Ok((f32::NEG_INFINITY, f32::INFINITY));
    }

    let lower = certified_affine_sum_f32_with_poll(
        0.0,
        objective
            .iter()
            .zip(output.lower().iter())
            .zip(output.upper().iter())
            .map(|((&coefficient, &lower), &upper)| {
                let endpoint = if coefficient >= 0.0 { lower } else { upper };
                (coefficient, endpoint)
            }),
        OutwardDirection::Lower,
        &mut poll,
    )?;
    let upper = certified_affine_sum_f32_with_poll(
        0.0,
        objective
            .iter()
            .zip(output.lower().iter())
            .zip(output.upper().iter())
            .map(|((&coefficient, &lower), &upper)| {
                let endpoint = if coefficient >= 0.0 { upper } else { lower };
                (coefficient, endpoint)
            }),
        OutwardDirection::Upper,
        &mut poll,
    )?;
    if lower.is_nan() || upper.is_nan() || lower > upper {
        return Ok((f32::NEG_INFINITY, f32::INFINITY));
    }

    let lower = ny_core::f64_to_f32_down(lower);
    let upper = ny_core::f64_to_f32_up(upper);
    if lower.is_nan() || upper.is_nan() || lower > upper {
        Ok((f32::NEG_INFINITY, f32::INFINITY))
    } else {
        Ok((lower, upper))
    }
}

fn finite_fallback_checkpoint(deadline: Instant, stage: &'static str) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(format!(
            "multi-objective certified fallback deadline exceeded {stage}"
        )));
    }
    Ok(())
}

pub(in crate::beta_crown::engine::graph) fn objective_bounds_multi_with_deadline(
    output: &BoundedTensor,
    objectives: &[Vec<f32>],
    deadline: Instant,
) -> Result<Vec<(f32, f32)>> {
    const SITE: &str = "multi-objective certified fallback bounds";
    const POLL_GRANULARITY: usize = 1024;

    finite_fallback_checkpoint(deadline, "before objective projection")?;
    let required_bytes = objectives
        .len()
        .checked_mul(size_of::<(f32, f32)>())
        .ok_or(NyError::CpuMemoryExceeded {
            required_bytes: usize::MAX,
            budget_bytes: crate::network::crown_memory::cpu_crown_dense_budget_bytes(),
            site: SITE,
        })?;
    let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    if required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: SITE,
        });
    }

    let mut bounds = Vec::new();
    bounds
        .try_reserve_exact(objectives.len())
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: SITE,
        })?;
    let mut units_since_poll = 0usize;
    for objective in objectives {
        finite_fallback_checkpoint(deadline, "before objective row")?;
        let bound = objective_bounds_with_poll(output, objective, |units| {
            units_since_poll = units_since_poll.saturating_add(units);
            if units_since_poll >= POLL_GRANULARITY {
                units_since_poll = 0;
                finite_fallback_checkpoint(deadline, "during objective row")?;
            }
            Ok(())
        })?;
        finite_fallback_checkpoint(deadline, "after objective row")?;
        bounds.push(bound);
    }
    Ok(bounds)
}

pub(in crate::beta_crown::engine::graph) fn clone_arc_node_bounds_with_deadline<'a>(
    source: impl Into<NodeBoundsView<'a>>,
    deadline: Instant,
) -> Result<HashMap<String, Arc<BoundedTensor>>> {
    const SITE: &str = "multi-objective certified fallback node cache";
    const POLL_GRANULARITY: usize = 256;

    finite_fallback_checkpoint(deadline, "before node-cache admission")?;
    let source = source.into();
    let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    let entry_bytes =
        size_of::<(String, Arc<BoundedTensor>)>().saturating_add(2 * size_of::<usize>());
    let mut required_bytes = source.len().saturating_mul(entry_bytes);
    for (index, name) in source.keys().enumerate() {
        required_bytes = required_bytes.saturating_add(name.len());
        if index.is_multiple_of(POLL_GRANULARITY) {
            finite_fallback_checkpoint(deadline, "during node-cache admission")?;
        }
    }
    if required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: SITE,
        });
    }

    let mut cloned = HashMap::new();
    cloned
        .try_reserve(source.len())
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: SITE,
        })?;
    finite_fallback_checkpoint(deadline, "after node-cache reserve")?;
    for (index, (name, bounds)) in source.iter().enumerate() {
        let mut cloned_name = String::new();
        cloned_name
            .try_reserve_exact(name.len())
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: SITE,
            })?;
        cloned_name.push_str(name);
        cloned.insert(cloned_name, Arc::clone(bounds));
        if index.is_multiple_of(POLL_GRANULARITY) {
            finite_fallback_checkpoint(deadline, "during node-cache clone")?;
        }
    }
    finite_fallback_checkpoint(deadline, "before node-cache publication")?;
    Ok(cloned)
}

fn build_spec_matrix(objectives: &[Vec<f32>]) -> Option<ndarray::Array2<f32>> {
    if objectives.is_empty() {
        return None;
    }
    let num_specs = objectives.len();
    let output_dim = objectives[0].len();
    let mut data = Vec::with_capacity(num_specs * output_dim);
    for objective in objectives {
        if objective.len() != output_dim {
            return None;
        }
        data.extend_from_slice(objective);
    }
    ndarray::Array2::from_shape_vec((num_specs, output_dim), data).ok()
}

pub(in crate::beta_crown::engine::graph) fn build_spec_matrix_for_authority(
    objectives: &[Vec<f32>],
    deadline: Option<Instant>,
) -> Result<ndarray::Array2<f32>> {
    let Some(deadline) = deadline else {
        return build_spec_matrix(objectives).ok_or_else(|| {
            NyError::InvalidSpec(
                "multi-objective dense spec matrix must be non-empty and rectangular".to_string(),
            )
        });
    };
    const SITE: &str = "multi-objective finite dense spec matrix";
    const POLL_GRANULARITY: usize = 1024;

    finite_fallback_checkpoint(deadline, "before dense spec admission")?;
    let Some(first) = objectives.first() else {
        return Err(NyError::InvalidSpec(
            "multi-objective dense spec matrix must be non-empty and rectangular".to_string(),
        ));
    };
    let rows = objectives.len();
    let columns = first.len();
    let elements = rows
        .checked_mul(columns)
        .ok_or(NyError::CpuMemoryExceeded {
            required_bytes: usize::MAX,
            budget_bytes: crate::network::crown_memory::cpu_crown_dense_budget_bytes(),
            site: SITE,
        })?;
    let required_bytes =
        elements
            .checked_mul(size_of::<f32>())
            .ok_or(NyError::CpuMemoryExceeded {
                required_bytes: usize::MAX,
                budget_bytes: crate::network::crown_memory::cpu_crown_dense_budget_bytes(),
                site: SITE,
            })?;
    let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    if required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: SITE,
        });
    }
    let mut data = Vec::new();
    data.try_reserve_exact(elements)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: SITE,
        })?;
    finite_fallback_checkpoint(deadline, "after dense spec reserve")?;
    let mut copied_since_poll = 0usize;
    for objective in objectives {
        if objective.len() != columns {
            return Err(NyError::InvalidSpec(
                "multi-objective dense spec matrix must be non-empty and rectangular".to_string(),
            ));
        }
        for &coefficient in objective {
            data.push(coefficient);
            copied_since_poll += 1;
            if copied_since_poll == POLL_GRANULARITY {
                copied_since_poll = 0;
                finite_fallback_checkpoint(deadline, "during dense spec copy")?;
            }
        }
    }
    finite_fallback_checkpoint(deadline, "before dense spec publication")?;
    ndarray::Array2::from_shape_vec((rows, columns), data).map_err(|_| {
        NyError::InternalError(
            "validated multi-objective dense spec shape was rejected".to_string(),
        )
    })
}

fn spec_bounds_to_vec(bounds: &BoundedTensor) -> Vec<(f32, f32)> {
    let flat = bounds.flatten();
    (0..flat.len())
        .map(|idx| (flat.lower()[[idx]], flat.upper()[[idx]]))
        .collect()
}

/// Dark gate: row-wise (per-spec) best-lower-bound merge across the
/// multi-objective β-optimization iterates.
///
/// Only the exact value `"1"` enables the experiment. Unset, `"0"`, `"true"`,
/// `" 1 "` and every other malformed value keep the legacy scalar-best path,
/// which is then byte-identical to today.
fn parse_mo_rowwise_merge(raw: Option<&str>) -> bool {
    raw == Some("1")
}

fn mo_rowwise_merge_enabled() -> bool {
    let raw = std::env::var("NY_MO_ROWWISE_MERGE").ok();
    parse_mo_rowwise_merge(raw.as_deref())
}

/// Default-dark baseline-first policy for deadline-bounded analytical
/// multi-objective β optimization.
///
/// Exact `1` is the only enabling spelling, and a finite effective graph-BaB
/// deadline is mandatory. Without a deadline there is no timeout result for a
/// completed baseline to protect, so spending an extra propagation pass would
/// only add work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MoBetaBaselineFirstPolicy {
    enabled: bool,
    /// Deterministic test seam: model the between-iteration deadline becoming
    /// due immediately before this iteration. Production policies leave this
    /// unset; tests use zero after a real baseline or one after one completed
    /// gradient pass.
    force_deadline_before_iteration: Option<usize>,
}

impl MoBetaBaselineFirstPolicy {
    fn from_raw(raw: Option<&str>, has_effective_deadline: bool) -> Self {
        Self {
            enabled: has_effective_deadline && raw == Some("1"),
            force_deadline_before_iteration: None,
        }
    }

    fn from_environment(has_effective_deadline: bool) -> Self {
        let raw = std::env::var("NY_MO_BETA_BASELINE_FIRST").ok();
        Self::from_raw(raw.as_deref(), has_effective_deadline)
    }

    #[cfg(test)]
    fn enabled_with_forced_iteration_zero_deadline() -> Self {
        Self {
            enabled: true,
            force_deadline_before_iteration: Some(0),
        }
    }

    #[cfg(test)]
    fn forced_deadline_after_first_completed_iteration() -> Self {
        Self {
            enabled: false,
            force_deadline_before_iteration: Some(1),
        }
    }

    fn deadline_reached_before_iteration(self, iteration: usize, actual: bool) -> bool {
        actual || self.force_deadline_before_iteration == Some(iteration)
    }
}

/// Default-dark bounded-baseline-only policy for analytical multi-objective
/// β optimization.
///
/// Exact `1` is the only enabling spelling, and a finite effective graph-BaB
/// deadline is mandatory. When enabled, the optimizer publishes one certified
/// Standard spec-guided pass and returns without entering the
/// intermediate-capturing gradient path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MoBetaBaselineOnlyPolicy {
    enabled: bool,
}

impl MoBetaBaselineOnlyPolicy {
    fn from_raw(raw: Option<&str>, has_effective_deadline: bool) -> Self {
        Self {
            enabled: has_effective_deadline && raw == Some("1"),
        }
    }

    fn from_environment(has_effective_deadline: bool) -> Self {
        let raw = std::env::var("NY_MO_BETA_BASELINE_ONLY").ok();
        Self::from_raw(raw.as_deref(), has_effective_deadline)
    }
}

/// Default-dark, deadline-bounded CUDA SPSA policy for analytical
/// multi-objective β optimization.
///
/// The optimizer implementation applies the remaining structural/backend
/// eligibility checks. Keeping this parser deliberately exact guarantees that
/// unset and malformed values retain the legacy path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MoCudaBetaSpsaPolicy {
    enabled: bool,
}

/// Completed-domain proof frontier presented to the optional sequential CUDA
/// β-SPSA lane.
///
/// `Empty` means there is genuinely no other completed open domain and admits
/// the first eligible child. `Invalid` is deliberately distinct: malformed or
/// non-finite queue metadata must fail closed instead of masquerading as an
/// empty frontier and authorizing advisory work.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(in crate::beta_crown::engine::graph) enum MoCudaBetaSpsaFrontier {
    #[default]
    Empty,
    Finite(f32),
    Invalid,
}

impl MoCudaBetaSpsaPolicy {
    fn from_raw(raw: Option<&str>, has_effective_deadline: bool) -> Self {
        Self {
            enabled: has_effective_deadline && raw == Some("1"),
        }
    }

    fn from_environment(has_effective_deadline: bool) -> Self {
        let raw = std::env::var("NY_MO_CUDA_BETA_SPSA").ok();
        Self::from_raw(raw.as_deref(), has_effective_deadline)
    }
}

pub(in crate::beta_crown::engine::graph) fn mo_cuda_beta_spsa_frontier_tracking_enabled(
    has_effective_deadline: bool,
) -> bool {
    MoCudaBetaSpsaPolicy::from_environment(has_effective_deadline).enabled
}

#[cfg(test)]
std::thread_local! {
    static MO_BETA_GRADIENT_PASS_COUNT: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static MO_BETA_COMPLETED_SPEC_PASS_COUNT: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn record_mo_beta_gradient_pass() {
    MO_BETA_GRADIENT_PASS_COUNT.with(|count| count.set(count.get() + 1));
}

#[cfg(test)]
fn record_mo_beta_completed_spec_pass() {
    MO_BETA_COMPLETED_SPEC_PASS_COUNT.with(|count| count.set(count.get() + 1));
}

#[cfg(test)]
pub(in crate::beta_crown::engine::graph) fn reset_mo_beta_gradient_pass_count_for_test() {
    MO_BETA_GRADIENT_PASS_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(in crate::beta_crown::engine::graph) fn mo_beta_gradient_pass_count_for_test() -> usize {
    MO_BETA_GRADIENT_PASS_COUNT.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(in crate::beta_crown::engine::graph) fn reset_mo_beta_completed_spec_pass_count_for_test() {
    MO_BETA_COMPLETED_SPEC_PASS_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(in crate::beta_crown::engine::graph) fn mo_beta_completed_spec_pass_count_for_test() -> usize {
    MO_BETA_COMPLETED_SPEC_PASS_COUNT.with(std::cell::Cell::get)
}

/// Fold one iterate's spec bounds into the running row-wise best lower bound.
///
/// SOUNDNESS (sound by construction): every iterate inside
/// `optimize_graph_beta_analytical_multi_objective_with_cache` evaluates the
/// SAME graph over the SAME input domain and the SAME dense spec matrix
/// (`build_spec_matrix(targets.objectives)`), so row `s` of every iterate is a
/// bound on the same linear functional `objectives[s]^T y`. Each iterate uses a
/// different β, and ANY `β >= 0` is a valid Lagrangian multiplier for the
/// active split constraints — so each iterate's row-`s` lower bound is an
/// *independently valid* lower bound on that row. Therefore the element-wise
/// maximum over iterates is itself a valid lower bound for each row. This is a
/// per-row max only: it never mixes rows across different spec orderings, and
/// it is never applied across different domains (the domain is fixed for the
/// whole call).
///
/// Non-finite lower bounds are skipped rather than folded: `-inf` carries no
/// information and `+inf`/`NaN` come from degenerate CROWN propagation (#2359)
/// and must not be allowed to poison a row.
///
/// Note the merged lower bound can never cross an upper bound: if `l_i <= v <=
/// u_j` holds for every iterate `i`, `j` on row `s`, then `max_i l_i <= u_j`.
fn fold_rowwise_lower(best_lo: &mut Option<Vec<f32>>, bounds: &[(f32, f32)]) {
    match best_lo {
        Some(lo) if lo.len() == bounds.len() => {
            for (s, b) in bounds.iter().enumerate() {
                if b.0.is_finite() && b.0 > lo[s] {
                    lo[s] = b.0;
                }
            }
        }
        // Defensive: a differing row count would mean a different spec
        // ordering, which we must never fold across. Skip.
        Some(_) => {}
        None => {
            *best_lo = Some(
                bounds
                    .iter()
                    .map(|b| {
                        if b.0.is_finite() {
                            b.0
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect(),
            );
        }
    }
}

/// Apply the row-wise best lower bound to a returned bound vector, tightening
/// each row that some earlier iterate bounded better. Upper bounds are left
/// untouched. See `fold_rowwise_lower` for the soundness argument.
fn apply_rowwise_lower(bounds: &mut [(f32, f32)], best_lo: &Option<Vec<f32>>) {
    let Some(lo) = best_lo else {
        return;
    };
    if lo.len() != bounds.len() {
        return;
    }
    for (s, b) in bounds.iter_mut().enumerate() {
        // `>` is false when `b.0` is NaN, so degenerate rows stay untouched.
        if lo[s].is_finite() && lo[s] > b.0 {
            b.0 = lo[s];
        }
    }
}

fn split_captured_multi_row_cache(
    captured_cache: Option<CachedLinearBounds>,
    num_objectives: usize,
) -> Vec<Option<CachedLinearBounds>> {
    captured_cache
        .and_then(|cache| cache.split_multi_row(num_objectives))
        .map(|per_objective| per_objective.into_iter().map(Some).collect())
        .unwrap_or_else(|| vec![None; num_objectives])
}

fn certified_inherited_multi_objective_fallback(
    graph: &GraphNetwork,
    context: &GraphCrownContext<'_>,
    objectives: &[Vec<f32>],
    deadline: Instant,
) -> Option<Result<MultiObjectiveWarmStartResult>> {
    const CACHE_SITE: &str = "multi-objective certified fallback cache slots";

    let base_bounds = context.base_bounds?;
    let result = (|| {
        finite_fallback_checkpoint(deadline, "before inherited-output lookup")?;
        let output_name = if graph.output_name().is_empty() {
            graph
                .exec_order()?
                .last()
                .map(String::as_str)
                .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            graph.output_name()
        };
        let output = base_bounds.get(output_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "multi-objective certified fallback missing output bounds for '{output_name}'"
            ))
        })?;
        let objective_bounds = objective_bounds_multi_with_deadline(output, objectives, deadline)?;
        let node_bounds = clone_arc_node_bounds_with_deadline(base_bounds, deadline)?;

        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        let required_bytes = objectives
            .len()
            .checked_mul(size_of::<Option<CachedLinearBounds>>())
            .ok_or(NyError::CpuMemoryExceeded {
                required_bytes: usize::MAX,
                budget_bytes,
                site: CACHE_SITE,
            })?;
        if required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: CACHE_SITE,
            });
        }
        let mut caches = Vec::new();
        caches
            .try_reserve_exact(objectives.len())
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: CACHE_SITE,
            })?;
        caches.resize_with(objectives.len(), || None);
        finite_fallback_checkpoint(deadline, "before inherited-bound publication")?;
        Ok((objective_bounds, node_bounds, caches))
    })();
    Some(result)
}

impl BetaCrownVerifier {
    /// Compute bounds for multiple objectives from output tensor using interval arithmetic.
    ///
    /// **Note**: This uses post-hoc interval arithmetic which loses output correlations.
    /// For tighter bounds, use spec-guided CROWN via `propagate_multi_objective_spec_guided`.
    pub(super) fn objective_bounds_multi(
        output: &BoundedTensor,
        objectives: &[Vec<f32>],
    ) -> Result<Vec<(f32, f32)>> {
        objectives
            .iter()
            .map(|obj| objective_bounds(output, obj))
            .collect()
    }

    /// Propagate bounds with β for multi-objective verification without optimization.
    ///
    /// This is used for deep domains where we skip β optimization and rely on
    /// inherited β values from warmup.
    ///
    /// Uses **spec-guided CROWN** with a dense multi-row spec matrix to preserve
    /// output correlations across all objectives in one backward pass, resulting
    /// in tighter bounds compared to post-hoc interval arithmetic.
    /// See issue #593 and docs/ANALYSIS-resnet-verification-gap-root-cause-2026-01-07.md.
    #[cfg(test)]
    pub(super) fn propagate_multi_objective_with_beta(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &GraphBetaState,
        targets: &MultiObjectiveTargets<'_>,
    ) -> Result<MultiObjectiveResult> {
        let seed_caches = vec![None; targets.objectives.len()];
        let (obj_bounds, final_node_bounds, _cached_las) = self
            .propagate_multi_objective_with_beta_and_cache(
                graph,
                input,
                context,
                beta_state,
                targets,
                &seed_caches,
                false,
            )?;
        Ok((obj_bounds, final_node_bounds))
    }

    // Justification: this helper threads graph context, beta state, objective
    // set, optional warm-start caches, and the capture flag together.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn propagate_multi_objective_with_beta_and_cache(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &GraphBetaState,
        targets: &MultiObjectiveTargets<'_>,
        seed_caches: &[Option<&CachedLinearBounds>],
        capture_caches: bool,
    ) -> Result<MultiObjectiveWarmStartResult> {
        if seed_caches.len() != targets.objectives.len() {
            return Err(NyError::InvalidSpec(format!(
                "multi-objective seed cache length {} != objective length {} (#3813)",
                seed_caches.len(),
                targets.objectives.len()
            )));
        }

        let spec_matrix = build_spec_matrix_for_authority(
            targets.objectives,
            self.effective_graph_bab_deadline(),
        )?;
        let combined_seed_cache = seed_caches
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
            .and_then(|caches| CachedLinearBounds::stack_single_row(&caches));
        let propagated = self.propagate_crown_with_graph_beta_and_spec_matrix(
            graph,
            input,
            context,
            beta_state,
            &spec_matrix,
            combined_seed_cache.as_ref(),
            capture_caches,
        );
        let (output, node_bounds, captured_cache) = match propagated {
            Ok(result) => result,
            Err(error)
                if super::is_finite_constrained_crown_refusal(&error)
                    && self.effective_graph_bab_deadline().is_some() =>
            {
                if let Some(fallback) = certified_inherited_multi_objective_fallback(
                    graph,
                    context,
                    targets.objectives,
                    self.effective_graph_bab_deadline()
                        .expect("finite refusal guard established a deadline"),
                ) {
                    tracing::debug!(
                        %error,
                        "Finite constrained multi-objective CROWN declined; retaining inherited certified bounds"
                    );
                    return fallback;
                }
                return Err(error);
            }
            Err(error) => return Err(error),
        };

        Ok((
            spec_bounds_to_vec(&output),
            node_bounds,
            split_captured_multi_row_cache(captured_cache, targets.objectives.len()),
        ))
    }

    /// Optimize β parameters using analytical gradients for multi-objective verification.
    ///
    /// Computes gradients analytically from the A matrices, avoiding the 3 forward
    /// passes per iteration that SPSA requires (~3x faster).
    ///
    /// For each iteration:
    /// 1. Propagate bounds and capture A matrices (1 forward pass)
    /// 2. Compute objective bounds for all objectives
    /// 3. Find the critical objective (min or max margin among unverified)
    /// 4. Compute β gradients for the critical objective using A matrices
    /// 5. Adam gradient step
    ///
    /// When `conjunctive` is true, optimizes the **maximum** margin instead of minimum.
    /// For conjunctive (AND) properties, only ONE objective needs to exceed its threshold
    /// to verify the domain, so we optimize the best objective rather than the worst.
    /// Source: designs/2026-03-05-joint-conjunctive-bab.md, #3334.
    #[cfg(test)]
    pub(super) fn optimize_graph_beta_analytical_multi_objective(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &mut GraphBetaState,
        targets: &MultiObjectiveTargets<'_>,
        conjunctive: bool,
    ) -> Result<MultiObjectiveResult> {
        let seed_caches = vec![None; targets.objectives.len()];
        let (obj_bounds, node_bounds, _cached_las) = self
            .optimize_graph_beta_analytical_multi_objective_with_cache(
                graph,
                input,
                context,
                beta_state,
                targets,
                conjunctive,
                &seed_caches,
                false,
            )?;
        Ok((obj_bounds, node_bounds))
    }

    // Justification: analytical beta optimization needs graph/input/context,
    // mutable beta state, multi-objective targets, conjunctive mode, and cache controls.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn optimize_graph_beta_analytical_multi_objective_with_cache(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &mut GraphBetaState,
        targets: &MultiObjectiveTargets<'_>,
        conjunctive: bool,
        seed_caches: &[Option<&CachedLinearBounds>],
        capture_caches: bool,
    ) -> Result<MultiObjectiveWarmStartResult> {
        let has_effective_deadline = self.effective_graph_bab_deadline().is_some();
        let baseline_first = MoBetaBaselineFirstPolicy::from_environment(has_effective_deadline);
        let baseline_only = MoBetaBaselineOnlyPolicy::from_environment(has_effective_deadline);
        self.optimize_graph_beta_analytical_multi_objective_with_cache_policy(
            graph,
            input,
            context,
            beta_state,
            targets,
            conjunctive,
            seed_caches,
            capture_caches,
            baseline_first,
            baseline_only,
            MoCudaBetaSpsaPolicy::default(),
            MoCudaBetaSpsaFrontier::Empty,
        )
    }

    /// Sequential-child optimizer entry with a deterministic completed-queue
    /// frontier for the optional CUDA β-SPSA follow-up.
    ///
    /// Keeping this as a distinct entry point prevents Rayon/batched callers
    /// from coupling SPSA work to scheduler timing. `Empty` means that no other
    /// completed open domain exists yet; `Invalid` fails closed and cannot be
    /// confused with that admitting empty-frontier state.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn optimize_graph_beta_analytical_multi_objective_with_cache_at_frontier(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &mut GraphBetaState,
        targets: &MultiObjectiveTargets<'_>,
        conjunctive: bool,
        seed_caches: &[Option<&CachedLinearBounds>],
        capture_caches: bool,
        cuda_beta_spsa_frontier: MoCudaBetaSpsaFrontier,
    ) -> Result<MultiObjectiveWarmStartResult> {
        let has_effective_deadline = self.effective_graph_bab_deadline().is_some();
        let baseline_first = MoBetaBaselineFirstPolicy::from_environment(has_effective_deadline);
        let baseline_only = MoBetaBaselineOnlyPolicy::from_environment(has_effective_deadline);
        let cuda_beta_spsa = MoCudaBetaSpsaPolicy::from_environment(has_effective_deadline);
        self.optimize_graph_beta_analytical_multi_objective_with_cache_policy(
            graph,
            input,
            context,
            beta_state,
            targets,
            conjunctive,
            seed_caches,
            capture_caches,
            baseline_first,
            baseline_only,
            cuda_beta_spsa,
            cuda_beta_spsa_frontier,
        )
    }

    /// Test-only deterministic deadline seam for the baseline-first policy.
    ///
    /// This runs the real Standard baseline propagation, then makes the existing
    /// between-iteration deadline guard fire immediately before iteration zero.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn optimize_graph_beta_analytical_multi_objective_with_cache_forced_baseline_deadline(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &mut GraphBetaState,
        targets: &MultiObjectiveTargets<'_>,
        conjunctive: bool,
        seed_caches: &[Option<&CachedLinearBounds>],
        capture_caches: bool,
    ) -> Result<MultiObjectiveWarmStartResult> {
        let baseline_first =
            MoBetaBaselineFirstPolicy::enabled_with_forced_iteration_zero_deadline();
        self.optimize_graph_beta_analytical_multi_objective_with_cache_policy(
            graph,
            input,
            context,
            beta_state,
            targets,
            conjunctive,
            seed_caches,
            capture_caches,
            baseline_first,
            MoBetaBaselineOnlyPolicy::default(),
            MoCudaBetaSpsaPolicy::default(),
            MoCudaBetaSpsaFrontier::Empty,
        )
    }

    /// Test-only seam that completes one ordinary no-deadline analytical pass,
    /// then stops at the next between-iteration guard.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn optimize_graph_beta_analytical_multi_objective_with_cache_forced_deadline_after_first_pass(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &mut GraphBetaState,
        targets: &MultiObjectiveTargets<'_>,
        conjunctive: bool,
        seed_caches: &[Option<&CachedLinearBounds>],
        capture_caches: bool,
    ) -> Result<MultiObjectiveWarmStartResult> {
        self.optimize_graph_beta_analytical_multi_objective_with_cache_policy(
            graph,
            input,
            context,
            beta_state,
            targets,
            conjunctive,
            seed_caches,
            capture_caches,
            MoBetaBaselineFirstPolicy::forced_deadline_after_first_completed_iteration(),
            MoBetaBaselineOnlyPolicy::default(),
            MoCudaBetaSpsaPolicy::default(),
            MoCudaBetaSpsaFrontier::Empty,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn optimize_graph_beta_analytical_multi_objective_with_cache_policy(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &mut GraphBetaState,
        targets: &MultiObjectiveTargets<'_>,
        conjunctive: bool,
        seed_caches: &[Option<&CachedLinearBounds>],
        capture_caches: bool,
        baseline_first: MoBetaBaselineFirstPolicy,
        baseline_only: MoBetaBaselineOnlyPolicy,
        cuda_beta_spsa: MoCudaBetaSpsaPolicy,
        cuda_beta_spsa_frontier: MoCudaBetaSpsaFrontier,
    ) -> Result<MultiObjectiveWarmStartResult> {
        // Skip if no beta parameters or iterations disabled
        if beta_state.is_empty() || self.config.beta_iterations == 0 {
            // Use spec-guided CROWN for tighter bounds (preserves output correlations)
            return self.propagate_multi_objective_with_beta_and_cache(
                graph,
                input,
                context,
                beta_state,
                targets,
                seed_caches,
                capture_caches,
            );
        }

        // Exact-gated finite-deadline lane: one independently certified
        // Standard propagation, then return before allocating or capturing the
        // analytical-gradient intermediates. `beta_state` is borrowed
        // immutably by the propagation, so the returned bounds and caller's β
        // state remain a consistent pair. Warm-start caches are intentionally
        // empty: capture is optional, and retaining them would add memory to the
        // bounded rescue path.
        if baseline_only.enabled {
            let (bounds, node_bounds, _discarded_caches) = self
                .propagate_multi_objective_with_beta_and_cache(
                    graph,
                    input,
                    context,
                    beta_state,
                    targets,
                    seed_caches,
                    false,
                )?;
            tracing::info!(
                objectives = targets.objectives.len(),
                "Multi-objective β optimization: baseline-only completed Standard pass"
            );
            return Ok((bounds, node_bounds, vec![None; targets.objectives.len()]));
        }

        // Baseline-only intentionally has precedence. The CUDA SPSA lane is an
        // optional child optimizer: `None` means that every preflight check was
        // side-effect free and the legacy algorithm/state path follows.
        if cuda_beta_spsa.enabled {
            if let Some(result) = self.try_optimize_multi_objective_cuda_beta_spsa(
                graph,
                input,
                context,
                beta_state,
                targets,
                conjunctive,
                seed_caches,
                cuda_beta_spsa_frontier,
            ) {
                return result;
            }
        }

        let spec_matrix = build_spec_matrix_for_authority(
            targets.objectives,
            self.effective_graph_bab_deadline(),
        )?;

        // Compute margin metric for optimization.
        // Disjunctive: min margin (all objectives must be verified → optimize worst).
        // Conjunctive: max margin (any objective verified suffices → optimize best). #3334
        // Defense-in-depth: assert lengths match before triple .zip() (#3383).
        debug_assert_eq!(
            targets.objectives.len(),
            targets.thresholds.len(),
            "compute_margin: objectives/thresholds length mismatch (#3383)"
        );
        debug_assert_eq!(
            targets.objectives.len(),
            targets.verified_mask.len(),
            "compute_margin: objectives/verified_mask length mismatch (#3383)"
        );
        let compute_margin = |bounds: &[(f32, f32)]| -> f32 {
            debug_assert_eq!(
                bounds.len(),
                targets.thresholds.len(),
                "compute_margin: bounds/thresholds length mismatch (#3383)"
            );
            let margins = bounds
                .iter()
                .zip(targets.thresholds.iter())
                .zip(targets.verified_mask.iter())
                .filter(|((_, _), &v)| !v) // Only unverified objectives
                .map(|(((l, _), &t), _)| l - t);
            if conjunctive {
                margins.fold(f32::NEG_INFINITY, nan_propagating_max)
            } else {
                margins.fold(f32::INFINITY, nan_propagating_min) // #2577: NaN margin must propagate
            }
        };

        let mut best_margin = f32::NEG_INFINITY;
        let mut best_beta_snapshot: Option<GraphBetaState> = None;

        // Dark gate `NY_MO_ROWWISE_MERGE=1`: keep a per-row (per-spec) best
        // lower bound across every iterate, not just the scalar-margin winner.
        // The scalar best discards a row that a *losing* iterate bounded better,
        // which bites depth-1..3 BaB children on multi-objective margin
        // properties. Every scalar `best_*` path below is left untouched, so
        // with the gate off this is byte-identical to the legacy behavior.
        let rowwise_merge = mo_rowwise_merge_enabled();
        let mut best_lo: Option<Vec<f32>> = None;

        // Best fully-computed spec result captured alongside its β snapshot.
        // On a mid-loop deadline we return THIS directly instead of running the
        // (expensive) final spec-guided pass + snapshot evaluation below. These
        // bounds come from `propagate_crown_with_graph_beta_and_spec_matrix_*`
        // over the same dense spec matrix as the final pass, so they are valid
        // sound spec-guided CROWN bounds. M5 may seed it from the equivalent
        // Standard helper before iteration zero; otherwise a completed gradient
        // iteration seeds it as before. We only stop optimizing sooner. (#3109)
        type LoopBest = (Vec<(f32, f32)>, HashMap<String, Arc<BoundedTensor>>);
        let mut best_loop_result: Option<LoopBest> = None;

        // Default-dark M5: secure one independently valid Standard spec-guided
        // child bound before the gradient-producing pass can consume the rest of
        // a finite deadline. This does not capture lA caches: the existing
        // deadline short-circuit returns empty cache slots, and cloning every
        // carrier solely to discard those caches would defeat the rescue path.
        //
        // `?` is intentional fail-closed behavior. If the baseline itself cannot
        // produce a certified bound, do not continue or manufacture a fallback.
        if baseline_first.enabled {
            let (baseline_bounds, baseline_node_bounds, _discarded_caches) = self
                .propagate_multi_objective_with_beta_and_cache(
                    graph,
                    input,
                    context,
                    beta_state,
                    targets,
                    seed_caches,
                    false,
                )?;
            let baseline_margin = compute_margin(&baseline_bounds);
            best_margin = if baseline_margin.is_nan() {
                f32::NEG_INFINITY
            } else {
                baseline_margin
            };
            best_beta_snapshot = Some(beta_state.clone());
            if rowwise_merge {
                fold_rowwise_lower(&mut best_lo, &baseline_bounds);
            }
            best_loop_result = Some((baseline_bounds, baseline_node_bounds));
        }

        // Periodic β snapshots for spec-guided evaluation at the end.
        // The post-hoc margin used during the loop is only a proxy and can be
        // β-insensitive for certain architectures (e.g., when objectives use
        // negative output coefficients whose lower bound depends on the output
        // UPPER relaxation, which β doesn't improve). Saving periodic snapshots
        // ensures we evaluate the optimal β that post-hoc tracking may miss.
        // At most 4 periodic snapshots to bound overhead. (#3334)
        let snapshot_interval = (self.config.beta_iterations / 4).max(1);
        let mut periodic_snapshots: Vec<GraphBetaState> = Vec::new();

        // Set when the loop bails because the wall-clock deadline was reached
        // (either the between-iteration guard or the inner per-node abort). Drives
        // the post-loop short-circuit that returns the best loop bounds. (#3109)
        let mut hit_deadline = false;
        // A typed finite-kernel refusal after a completed iterate must retain
        // that stronger result rather than replacing it with inherited IBP.
        let mut retain_completed_best = false;

        for iter in 0..self.config.beta_iterations {
            // Deadline check (#3109): bail early if the verification timeout budget
            // is exhausted BETWEEN iterations. Return the current best bounds
            // instead of running all iterations. This is the cheap guard; the
            // inner per-node deadline check below is what usually fires first.
            if baseline_first
                .deadline_reached_before_iteration(iter, self.past_effective_graph_bab_deadline())
            {
                tracing::info!(
                    "Multi-objective β optimization: deadline exceeded at iteration {}/{}, returning best bounds",
                    iter, self.config.beta_iterations
                );
                hit_deadline = true;
                break;
            }

            // Reset gradients
            beta_state.zero_grad();

            #[cfg(test)]
            record_mo_beta_gradient_pass();

            // Compute bounds with current β AND capture per-objective A rows in one pass.
            //
            // Deadline granularity (#3109): a single spec-guided backward pass over
            // ~99-199 specs in a deep Conv2d graph can take seconds, and the inner
            // per-node deadline check (constraints/backward/mod.rs) aborts that pass
            // mid-flight with `DeadlineExceeded` once the wall clock crosses the
            // budget. Previously the `?` here propagated that error straight out,
            // DISCARDING every completed beta-opt iteration's bounds. Instead, if we
            // already have a best fully-computed result from an earlier iteration,
            // break and return it via the post-loop deadline short-circuit below.
            // Returning that earlier (valid) spec-guided CROWN bound is sound — we
            // only stop optimizing sooner and yield to BaB/timeout gracefully.
            let (output, node_bounds, intermediate) = match self
                .propagate_crown_with_graph_beta_and_spec_matrix_storing_intermediates(
                    graph,
                    input,
                    context,
                    beta_state,
                    &spec_matrix,
                ) {
                Ok(triple) => triple,
                Err(e) if e.is_deadline_exceeded() && best_loop_result.is_some() => {
                    tracing::info!(
                        "Multi-objective β optimization: inner pass hit deadline at iteration \
                         {}/{}, returning best completed spec bounds",
                        iter,
                        self.config.beta_iterations
                    );
                    hit_deadline = true;
                    break;
                }
                Err(error)
                    if super::is_finite_constrained_crown_refusal(&error)
                        && self.effective_graph_bab_deadline().is_some() =>
                {
                    if best_loop_result.is_some() {
                        tracing::debug!(
                            %error,
                            "Finite analytical β pass declined; retaining best completed iterate"
                        );
                        retain_completed_best = true;
                        break;
                    }
                    let deadline = self
                        .effective_graph_bab_deadline()
                        .expect("finite refusal guard established a deadline");
                    if let Some(fallback) = certified_inherited_multi_objective_fallback(
                        graph,
                        context,
                        targets.objectives,
                        deadline,
                    ) {
                        tracing::debug!(
                            %error,
                            "Finite analytical β pass declined; retaining inherited certified bounds"
                        );
                        return fallback;
                    }
                    return Err(error);
                }
                Err(e) => return Err(e),
            };

            #[cfg(test)]
            record_mo_beta_completed_spec_pass();
            let obj_bounds = spec_bounds_to_vec(&output);
            // Fold site 1: this iterate's spec-guided rows, same spec matrix
            // and same domain as sites 2 and 3.
            if rowwise_merge {
                fold_rowwise_lower(&mut best_lo, &obj_bounds);
            }
            let margin = compute_margin(&obj_bounds);

            // Track best β state by post-hoc margin. See #1694.
            // Also capture the full (bounds, node_bounds) so a mid-loop deadline
            // can return immediately without an extra spec-guided pass. (#3109)
            if margin > best_margin {
                best_margin = margin;
                best_beta_snapshot = Some(beta_state.clone());
                best_loop_result = Some((obj_bounds.clone(), node_bounds));
            }

            // Save periodic snapshot for spec-guided evaluation at end.
            // Post-hoc margin can be β-insensitive, so we save at fixed intervals
            // rather than only when margin improves. (#3334)
            if iter.is_multiple_of(snapshot_interval) {
                periodic_snapshots.push(beta_state.clone());
            }

            // Compute analytical gradients for the critical objective
            let max_grad = beta_state.compute_analytical_gradients_multi_objective_spec_rows(
                &intermediate,
                &obj_bounds,
                targets.thresholds,
                targets.verified_mask,
                conjunctive,
            );

            // Adam gradient step
            let t = iter + 1;
            beta_state.gradient_step_adam(&self.config.adaptive_config, t);

            // Check convergence
            if max_grad < self.config.beta_tolerance {
                trace!(
                    "Graph β-analytical multi-obj converged at iteration {} (max_grad={:.6})",
                    iter,
                    max_grad
                );
                break;
            }
        }

        // Deadline short-circuit (#3109): if the loop bailed on the wall-clock
        // deadline AND we captured a fully-computed baseline or iteration result,
        // do NOT run the final spec-guided pass or the per-candidate
        // snapshot evaluation below — each is a full spec-guided CROWN pass over
        // every objective, and for deep Conv2d graphs with ~99-199 specs a single
        // pass can itself overrun the remaining budget (and would in fact abort
        // mid-flight with `DeadlineExceeded`, discarding all the beta-opt work).
        // Instead return the best fully-computed bounds captured before/during the loop.
        // These are valid spec-guided CROWN bounds (sound); we just stop optimizing
        // sooner and yield to BaB/timeout gracefully. We return `None` caches:
        // warm-starting is an optimization, and skipping it is sound (the next
        // round simply recomputes). We also sync `beta_state` to the best snapshot
        // for a consistent (bounds, β) pair.
        let early_return =
            hit_deadline || retain_completed_best || self.past_effective_graph_bab_deadline();
        if let Some((mut bounds, node_bounds)) = best_loop_result.filter(|_| early_return) {
            if let Some(best_beta) = best_beta_snapshot {
                *beta_state = best_beta;
            }
            // Gate-on only: tighten each row with the best lower bound any
            // completed baseline/iterate produced for that row. `node_bounds` is left as
            // captured — intermediate-layer bounds are sound on their own and
            // do not depend on which iterate supplied the spec-row lower bound.
            if rowwise_merge {
                apply_rowwise_lower(&mut bounds, &best_lo);
            }
            tracing::info!(
                "Multi-objective β optimization: deadline exceeded, returning best \
                 completed spec bounds without final spec-guided pass or snapshot evaluation"
            );
            let caches = vec![None; targets.objectives.len()];
            return Ok((bounds, node_bounds, caches));
        }
        // Otherwise (no deadline hit, or no baseline/iteration completed before
        // it), fall through to compute the final spec-guided bounds so callers
        // always get a valid result. With M5 dark, a deadline already elapsed
        // before iteration 0 makes this final pass abort with `DeadlineExceeded`,
        // preserving the historical graceful-timeout behavior.

        // Compute final spec-guided bounds with the end-of-loop β state.
        let (final_bounds, final_node_bounds, final_cached_las) = self
            .propagate_multi_objective_with_beta_and_cache(
                graph,
                input,
                context,
                beta_state,
                targets,
                seed_caches,
                capture_caches,
            )?;
        // Fold site 2: the final spec-guided pass uses the identical spec
        // matrix (`build_spec_matrix(targets.objectives)`) over the identical
        // domain, so its row `s` is the same functional as site 1's row `s`.
        if rowwise_merge {
            fold_rowwise_lower(&mut best_lo, &final_bounds);
        }
        let final_margin = compute_margin(&final_bounds);

        // Evaluate all snapshot candidates with spec-guided CROWN and return
        // the β state with the best spec-guided margin. This catches the
        // optimal β that post-hoc tracking may have missed. (#3334, #1694)
        let mut best_overall_margin = final_margin;
        let mut best_overall_bounds = final_bounds;
        let mut best_overall_node_bounds = final_node_bounds;
        let mut best_overall_cached_las = final_cached_las;
        let mut best_overall_beta: Option<GraphBetaState> = None;

        // Deadline guard (#3813): snapshot evaluation runs N spec-guided CROWN
        // passes per candidate (one per objective). For expensive Conv2d graphs
        // with many objectives, this can exceed the remaining timeout budget.
        // Skip snapshot evaluation entirely when past deadline — the final
        // spec-guided bounds above are the best we can return in time.
        if !self.past_effective_graph_bab_deadline() {
            let candidates = periodic_snapshots.into_iter().chain(best_beta_snapshot);
            for candidate in candidates {
                // Per-candidate deadline check: bail before each expensive
                // spec-guided evaluation to avoid overrunning the timeout.
                if self.past_effective_graph_bab_deadline() {
                    tracing::debug!(
                        "Multi-objective β snapshot evaluation: deadline exceeded, \
                         returning best bounds from {} evaluated candidates",
                        if best_overall_beta.is_some() {
                            "partial"
                        } else {
                            "final-only"
                        }
                    );
                    break;
                }
                let (bounds, node_bounds, cached_las) = self
                    .propagate_multi_objective_with_beta_and_cache(
                        graph,
                        input,
                        context,
                        &candidate,
                        targets,
                        seed_caches,
                        capture_caches,
                    )?;
                // Fold site 3: same helper, same spec matrix, same domain —
                // only β differs, so row `s` is again the same functional.
                if rowwise_merge {
                    fold_rowwise_lower(&mut best_lo, &bounds);
                }
                let margin = compute_margin(&bounds);
                // Strict > so that ties prefer the end-of-loop β (most recent
                // optimizer state for warm-starting downstream). (#1760)
                if margin > best_overall_margin {
                    best_overall_margin = margin;
                    best_overall_bounds = bounds;
                    best_overall_node_bounds = node_bounds;
                    best_overall_cached_las = cached_las;
                    best_overall_beta = Some(candidate);
                }
            }
        }

        // Update beta_state to match the returned bounds so callers have a
        // consistent (bounds, beta_state) pair for warm-starting. (#1760)
        if let Some(best_beta) = best_overall_beta {
            *beta_state = best_beta;
        }

        // Gate-on only: the scalar-margin winner above may still be looser on
        // individual rows than some other iterate was. Tighten row-wise.
        if rowwise_merge {
            apply_rowwise_lower(&mut best_overall_bounds, &best_lo);
        }

        Ok((
            best_overall_bounds,
            best_overall_node_bounds,
            best_overall_cached_las,
        ))
    }
}

#[cfg(test)]
mod objective_outward_rounding_tests {
    use super::objective_bounds;
    use ndarray::Array1;
    use ny_tensor::BoundedTensor;

    #[test]
    fn cancellation_cannot_make_root_objective_interval_inward() {
        let large = 2.0_f32.powi(30);
        let point = Array1::from_vec(vec![large, 1.0, large]).into_dyn();
        let output = BoundedTensor::new(point.clone(), point).expect("point box");
        let (lower, upper) =
            objective_bounds(&output, &[large, 1.0, -large]).expect("matching objective");

        // Exact binary value: 2^60 + 1 - 2^60 = 1. A nearest-f64 fold
        // loses the middle term and final one-ULP f32 widening cannot recover it.
        assert!(lower <= 1.0, "lower {lower:e} must enclose exact 1");
        assert!(upper >= 1.0, "upper {upper:e} must enclose exact 1");
        assert!(
            lower > 0.99 && upper < 1.01,
            "certified DD reduction must stay useful: [{lower:e}, {upper:e}]"
        );
    }
}

#[cfg(test)]
mod rowwise_merge_tests {
    use super::{
        apply_rowwise_lower, fold_rowwise_lower, parse_mo_rowwise_merge, MoBetaBaselineFirstPolicy,
        MoBetaBaselineOnlyPolicy, MoCudaBetaSpsaPolicy,
    };

    #[test]
    fn baseline_first_policy_arms_only_on_exact_one_with_deadline() {
        assert_eq!(
            MoBetaBaselineFirstPolicy::from_raw(Some("1"), true),
            MoBetaBaselineFirstPolicy {
                enabled: true,
                force_deadline_before_iteration: None,
            }
        );
        assert!(!MoBetaBaselineFirstPolicy::from_raw(Some("1"), false).enabled);
    }

    #[test]
    fn baseline_first_policy_rejects_every_other_spelling() {
        for raw in [
            None,
            Some("0"),
            Some("true"),
            Some("TRUE"),
            Some("yes"),
            Some(" 1 "),
            Some("1 "),
            Some(" 1"),
            Some("01"),
            Some("1.0"),
            Some(""),
            Some("on"),
            Some("-1"),
            Some("11"),
        ] {
            assert!(
                !MoBetaBaselineFirstPolicy::from_raw(raw, true).enabled,
                "baseline-first gate must not arm on {raw:?}"
            );
        }
    }

    #[test]
    fn baseline_only_policy_arms_only_on_exact_one_with_deadline() {
        assert_eq!(
            MoBetaBaselineOnlyPolicy::from_raw(Some("1"), true),
            MoBetaBaselineOnlyPolicy { enabled: true }
        );
        assert!(
            !MoBetaBaselineOnlyPolicy::from_raw(Some("1"), false).enabled,
            "an unbounded optimizer must retain the existing gradient path"
        );
    }

    #[test]
    fn baseline_only_policy_rejects_every_other_spelling() {
        for raw in [
            None,
            Some("0"),
            Some("true"),
            Some("TRUE"),
            Some("yes"),
            Some(" 1 "),
            Some("1 "),
            Some(" 1"),
            Some("01"),
            Some("1.0"),
            Some(""),
            Some("on"),
            Some("-1"),
            Some("11"),
        ] {
            assert!(
                !MoBetaBaselineOnlyPolicy::from_raw(raw, true).enabled,
                "baseline-only gate must not arm on {raw:?}"
            );
        }
    }

    #[test]
    fn cuda_beta_spsa_policy_arms_only_on_exact_one_with_deadline() {
        assert_eq!(
            MoCudaBetaSpsaPolicy::from_raw(Some("1"), true),
            MoCudaBetaSpsaPolicy { enabled: true }
        );
        assert!(!MoCudaBetaSpsaPolicy::from_raw(Some("1"), false).enabled);
    }

    #[test]
    fn cuda_beta_spsa_policy_rejects_every_other_spelling() {
        for raw in [
            None,
            Some("0"),
            Some("true"),
            Some("TRUE"),
            Some("yes"),
            Some(" 1 "),
            Some("1 "),
            Some(" 1"),
            Some("01"),
            Some("1.0"),
            Some(""),
            Some("on"),
            Some("-1"),
            Some("11"),
        ] {
            assert!(
                !MoCudaBetaSpsaPolicy::from_raw(raw, true).enabled,
                "CUDA β-SPSA gate must not arm on {raw:?}"
            );
        }
    }

    #[test]
    fn gate_arms_only_on_exact_one() {
        assert!(parse_mo_rowwise_merge(Some("1")));
    }

    #[test]
    fn gate_does_not_arm_on_malformed_values() {
        // Mandatory malformed-value coverage: unset, explicit zero, truthy
        // words, and whitespace-padded "1" must all keep the legacy path.
        for raw in [
            None,
            Some("0"),
            Some("true"),
            Some("TRUE"),
            Some("yes"),
            Some(" 1 "),
            Some("1 "),
            Some(" 1"),
            Some("01"),
            Some("1.0"),
            Some(""),
            Some("on"),
            Some("-1"),
            Some("11"),
        ] {
            assert!(!parse_mo_rowwise_merge(raw), "gate must not arm on {raw:?}");
        }
    }

    #[test]
    fn fold_initialises_from_first_vector_and_sanitises_non_finite() {
        let mut best: Option<Vec<f32>> = None;
        fold_rowwise_lower(
            &mut best,
            &[
                (1.0, 2.0),
                (f32::NEG_INFINITY, 5.0),
                (f32::NAN, 5.0),
                (f32::INFINITY, f32::INFINITY),
            ],
        );
        let lo = best.expect("initialised");
        assert_eq!(lo[0], 1.0);
        // Every non-finite lower (−inf, NaN, +inf) sanitises to −inf so it can
        // neither poison the row nor block a later finite improvement.
        assert_eq!(lo[1], f32::NEG_INFINITY);
        assert_eq!(lo[2], f32::NEG_INFINITY);
        assert_eq!(lo[3], f32::NEG_INFINITY);
    }

    #[test]
    fn fold_takes_row_wise_max_across_iterates() {
        let mut best: Option<Vec<f32>> = None;
        // Iterate A wins row 0, iterate B wins row 1: the scalar-best path
        // would keep only one of them.
        fold_rowwise_lower(&mut best, &[(3.0, 9.0), (0.5, 9.0)]);
        fold_rowwise_lower(&mut best, &[(1.0, 9.0), (4.0, 9.0)]);
        assert_eq!(best, Some(vec![3.0, 4.0]));
    }

    #[test]
    fn fold_skips_non_finite_and_shape_mismatch() {
        let mut best: Option<Vec<f32>> = Some(vec![3.0, 4.0]);
        fold_rowwise_lower(&mut best, &[(f32::INFINITY, 9.0), (f32::NAN, 9.0)]);
        assert_eq!(best, Some(vec![3.0, 4.0]));
        // A differing row count means a different spec ordering: never fold.
        fold_rowwise_lower(&mut best, &[(100.0, 9.0)]);
        assert_eq!(best, Some(vec![3.0, 4.0]));
    }

    #[test]
    fn apply_tightens_only_improved_rows_and_never_crosses_upper() {
        let mut bounds = [(1.0f32, 5.0f32), (4.5, 5.0), (f32::NAN, 5.0)];
        let best = Some(vec![3.0f32, 2.0, 3.0]);
        apply_rowwise_lower(&mut bounds, &best);
        assert_eq!(bounds[0].0, 3.0); // improved
        assert_eq!(bounds[0].1, 5.0); // upper untouched
        assert_eq!(bounds[1].0, 4.5); // already tighter, unchanged
        assert!(bounds[2].0.is_nan()); // degenerate row left alone
        for (l, u) in bounds.iter().take(2) {
            assert!(l <= u, "merged lower must not cross upper");
        }
    }

    #[test]
    fn apply_is_a_noop_without_a_merge_or_on_shape_mismatch() {
        let mut bounds = [(1.0f32, 5.0f32)];
        apply_rowwise_lower(&mut bounds, &None);
        assert_eq!(bounds[0], (1.0, 5.0));
        apply_rowwise_lower(&mut bounds, &Some(vec![9.0, 9.0]));
        assert_eq!(bounds[0], (1.0, 5.0));
    }
}
