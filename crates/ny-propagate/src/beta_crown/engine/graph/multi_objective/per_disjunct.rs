// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-disjunct alpha optimization for multi-objective graph BaB (#4355).
//!
//! When `optimize_disjuncts_separately` is enabled, each unverified disjunct
//! gets its own alpha state optimized to prove that specific output constraint.
//! Verified disjuncts inherit the shared root alpha to save optimization budget.
//!
//! Reference: alpha-beta-CROWN `beta_CROWN_solver.py:1098`

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, info};

use crate::beta_crown::state::GraphDomainAlphaState;
use crate::beta_crown::BetaCrownConfig;
use crate::bounds::GraphAlphaState;
use crate::AlphaCrownConfig;
use crate::GraphNetwork;

use super::super::shared::setup::build_root_alpha_state_from_root_alpha;

const PER_DISJUNCT_PHASE_FRACTION_DENOMINATOR: u32 = 4;
const PER_DISJUNCT_PHASE_MAX: Duration = Duration::from_secs(15);
const MIN_ITERATIONS_PER_DISJUNCT: usize = 2;

/// Give the per-disjunct optimizer a live deadline after the earlier warmup
/// phase has spent its private cap.
///
/// `GraphBabBootstrap::alpha_config` deliberately carries the warmup deadline.
/// Reusing an expired copy here makes `optimize_alpha_for_spec_objective` stop
/// before iteration zero. Per-disjunct mode is already an explicit, default-off
/// opt-in, so only that mode rebinds a missing or expired deadline to the
/// enclosing BaB deadline. A still-live warmup deadline remains unchanged
/// (never extended), and an exhausted BaB deadline remains exhausted.
fn rebind_missing_or_expired_alpha_deadline_for_bab(
    inherited: &AlphaCrownConfig,
    now: Instant,
    bab_deadline: Instant,
) -> AlphaCrownConfig {
    let mut rebound = inherited.clone();
    if inherited.deadline.is_none_or(|deadline| deadline <= now) && bab_deadline > now {
        rebound.deadline = Some(bab_deadline);
    }
    rebound
}

/// Bound the entire sequential per-disjunct phase to one quarter of the live
/// BaB remainder, capped at 15 seconds.
///
/// A still-live, earlier inherited deadline remains authoritative. Missing or
/// expired warmup deadlines are first rebound to the live BaB deadline. When
/// BaB is already exhausted, return its expired deadline so this optional phase
/// cannot start unbounded work.
fn per_disjunct_phase_deadline(
    inherited: &AlphaCrownConfig,
    now: Instant,
    bab_deadline: Instant,
) -> Instant {
    if bab_deadline <= now {
        return bab_deadline;
    }

    let rebound = rebind_missing_or_expired_alpha_deadline_for_bab(inherited, now, bab_deadline);
    let global_remaining = bab_deadline.duration_since(now);
    let phase_budget =
        (global_remaining / PER_DISJUNCT_PHASE_FRACTION_DENOMINATOR).min(PER_DISJUNCT_PHASE_MAX);
    let phase_cap = now + phase_budget;
    rebound.deadline.unwrap_or(bab_deadline).min(phase_cap)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PerDisjunctAllocation {
    iterations: usize,
    deadline: Instant,
    rows_remaining_after: usize,
    iterations_remaining_after: usize,
}

/// Fair scheduler for sequential row-specialized alpha optimization.
///
/// Iterations form a fixed total pool. Wall time is divided from the *current*
/// phase remainder on every allocation, so a row that finishes before its
/// deadline leaves that time available to the remaining rows.
struct PerDisjunctBudget {
    iterations_remaining: usize,
    rows_remaining: usize,
    phase_deadline: Instant,
}

impl PerDisjunctBudget {
    fn new(total_iterations: usize, rows: usize, phase_deadline: Instant) -> Option<Self> {
        if rows == 0 || total_iterations / rows < MIN_ITERATIONS_PER_DISJUNCT {
            return None;
        }
        Some(Self {
            iterations_remaining: total_iterations,
            rows_remaining: rows,
            phase_deadline,
        })
    }

    fn next(&mut self, now: Instant) -> Option<PerDisjunctAllocation> {
        if self.rows_remaining == 0 {
            return None;
        }

        let iterations = self.iterations_remaining / self.rows_remaining;
        let wall_remaining = self.phase_deadline.saturating_duration_since(now);
        let wall_share = wall_remaining.div_f64(self.rows_remaining as f64);
        let deadline = (now + wall_share).min(self.phase_deadline);

        self.iterations_remaining -= iterations;
        self.rows_remaining -= 1;
        Some(PerDisjunctAllocation {
            iterations,
            deadline,
            rows_remaining_after: self.rows_remaining,
            iterations_remaining_after: self.iterations_remaining,
        })
    }
}

/// Build per-disjunct alpha states by optimizing alpha independently for each
/// unverified disjunct.
///
/// Skips disjuncts that are already verified at root (lower > threshold).
/// For those, inherits the shared alpha. This focuses optimization budget
/// on the disjuncts that actually need tighter bounds.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_per_disjunct_alphas(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    root_alpha: &GraphAlphaState,
    alpha_config: &AlphaCrownConfig,
    bab_deadline: Instant,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    initial_obj_bounds: &[(f32, f32)],
    history: &crate::beta_crown::branching::GraphSplitHistory,
    initial_node_bounds_arc: &HashMap<String, Arc<BoundedTensor>>,
    engine: Option<&dyn GemmEngine>,
) -> Result<Vec<GraphDomainAlphaState>> {
    let num_disjuncts = objectives.len();
    if num_disjuncts == 0
        || thresholds.len() != num_disjuncts
        || initial_obj_bounds.len() != num_disjuncts
    {
        return Err(NyError::InvalidSpec(format!(
            "per-disjunct alpha layout mismatch: objectives={}, thresholds={}, bounds={}",
            num_disjuncts,
            thresholds.len(),
            initial_obj_bounds.len()
        )));
    }
    for (index, ((objective, &(lower, upper)), &threshold)) in objectives
        .iter()
        .zip(initial_obj_bounds)
        .zip(thresholds)
        .enumerate()
    {
        if objective.is_empty()
            || objective.iter().any(|value| !value.is_finite())
            || !lower.is_finite()
            || !upper.is_finite()
            || !threshold.is_finite()
            || lower > upper
        {
            return Err(NyError::NumericalInstability(format!(
                "per-disjunct alpha row {index} is malformed"
            )));
        }
    }
    let mut per_disjunct = Vec::with_capacity(num_disjuncts);

    // Convert Arc bounds to plain BoundedTensor for the alpha optimizer.
    let ibp_bounds: HashMap<String, BoundedTensor> = initial_node_bounds_arc
        .iter()
        .map(|(k, v)| (k.clone(), v.as_ref().clone()))
        .collect();

    // Shared domain alpha as fallback for already-verified disjuncts.
    let shared_domain_alpha = build_root_alpha_state_from_root_alpha(
        root_alpha,
        graph,
        input,
        history,
        initial_node_bounds_arc,
    );

    let unverified_count = objectives
        .iter()
        .enumerate()
        .filter(|(idx, _)| {
            !BetaCrownConfig::domain_is_verified_for_mode(
                false,
                initial_obj_bounds[*idx].0,
                initial_obj_bounds[*idx].1,
                thresholds[*idx],
            )
        })
        .count();

    if unverified_count == 0 {
        per_disjunct.resize(num_disjuncts, shared_domain_alpha);
        info!(
            "Per-disjunct alpha: all {num_disjuncts} disjuncts already verified; no specialization"
        );
        return Ok(per_disjunct);
    }

    let phase_start = Instant::now();
    let phase_deadline = per_disjunct_phase_deadline(alpha_config, phase_start, bab_deadline);
    let Some(mut budget) =
        PerDisjunctBudget::new(alpha_config.iterations, unverified_count, phase_deadline)
    else {
        per_disjunct.resize(num_disjuncts, shared_domain_alpha);
        info!(
            "Per-disjunct alpha: skipped specialization for {unverified_count} unverified rows; \
             total iteration budget {} cannot provide at least {MIN_ITERATIONS_PER_DISJUNCT} per row",
            alpha_config.iterations,
        );
        return Ok(per_disjunct);
    };

    info!(
        "Per-disjunct alpha budget: {unverified_count} unverified rows, {} total iterations, phase cap {:?}",
        alpha_config.iterations,
        phase_deadline.saturating_duration_since(phase_start),
    );

    let mut optimized_count = 0;
    let mut verified_count = 0;
    let mut failed_count = 0;
    let mut deadline_skipped_count = 0;
    let mut assigned_iterations = 0;
    for (idx, spec_row) in objectives.iter().enumerate() {
        let already_verified = BetaCrownConfig::domain_is_verified_for_mode(
            false,
            initial_obj_bounds[idx].0,
            initial_obj_bounds[idx].1,
            thresholds[idx],
        );

        if already_verified {
            per_disjunct.push(shared_domain_alpha.clone());
            verified_count += 1;
            continue;
        }

        let allocation_now = Instant::now();
        let allocation = budget
            .next(allocation_now)
            .expect("unverified row count must match scheduler row count");
        assigned_iterations += allocation.iterations;
        debug!(
            "Per-disjunct alpha row {idx}: {} iterations, wall share {:?}, {} rows / {} iterations remain",
            allocation.iterations,
            allocation.deadline.saturating_duration_since(allocation_now),
            allocation.rows_remaining_after,
            allocation.iterations_remaining_after,
        );

        if allocation.deadline <= allocation_now {
            deadline_skipped_count += 1;
            per_disjunct.push(shared_domain_alpha.clone());
            continue;
        }

        let mut row_config = alpha_config.clone();
        row_config.iterations = allocation.iterations;
        row_config.deadline = Some(allocation.deadline);

        // Optimize alpha targeting this specific disjunct's spec row.
        match graph.optimize_alpha_for_spec_objective(
            input,
            &ibp_bounds,
            root_alpha,
            &row_config,
            spec_row,
            engine,
        ) {
            Ok(optimized_alpha) => {
                let domain_alpha = build_root_alpha_state_from_root_alpha(
                    &optimized_alpha,
                    graph,
                    input,
                    history,
                    initial_node_bounds_arc,
                );
                per_disjunct.push(domain_alpha);
                optimized_count += 1;
            }
            Err(e) => {
                debug!(
                    "Per-disjunct alpha optimization failed for disjunct {}: {}, using shared alpha",
                    idx, e
                );
                failed_count += 1;
                per_disjunct.push(shared_domain_alpha.clone());
            }
        }
    }

    info!(
        "Per-disjunct alpha (#4355): optimized {optimized_count}/{num_disjuncts}; \
         verified={verified_count}, failed={failed_count}, deadline_skipped={deadline_skipped_count}, \
         assigned_iterations={assigned_iterations}/{}, elapsed={:?}",
        alpha_config.iterations,
        phase_start.elapsed(),
    );

    Ok(per_disjunct)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expired_warmup_deadline_rebinds_to_live_bab_deadline() {
        let now = Instant::now();
        let warmup_deadline = now
            .checked_sub(Duration::from_secs(1))
            .expect("test clock must permit a one-second offset");
        let bab_deadline = now + Duration::from_secs(20);
        let inherited = AlphaCrownConfig {
            iterations: 7,
            deadline: Some(warmup_deadline),
            ..Default::default()
        };

        let rebound =
            rebind_missing_or_expired_alpha_deadline_for_bab(&inherited, now, bab_deadline);

        assert_eq!(rebound.deadline, Some(bab_deadline));
        assert_eq!(rebound.iterations, 7, "only the phase deadline may change");
        assert_eq!(inherited.deadline, Some(warmup_deadline));
    }

    #[test]
    fn live_warmup_deadline_is_not_extended() {
        let now = Instant::now();
        let warmup_deadline = now + Duration::from_secs(3);
        let bab_deadline = now + Duration::from_secs(20);
        let inherited = AlphaCrownConfig {
            deadline: Some(warmup_deadline),
            ..Default::default()
        };

        let rebound =
            rebind_missing_or_expired_alpha_deadline_for_bab(&inherited, now, bab_deadline);

        assert_eq!(rebound.deadline, Some(warmup_deadline));
    }

    #[test]
    fn missing_warmup_deadline_binds_to_live_bab_deadline() {
        let now = Instant::now();
        let bab_deadline = now + Duration::from_secs(20);
        let inherited = AlphaCrownConfig {
            deadline: None,
            ..Default::default()
        };

        let rebound =
            rebind_missing_or_expired_alpha_deadline_for_bab(&inherited, now, bab_deadline);

        assert_eq!(rebound.deadline, Some(bab_deadline));
    }

    #[test]
    fn expired_bab_deadline_is_not_revived() {
        let now = Instant::now();
        let inherited = AlphaCrownConfig {
            deadline: Some(
                now.checked_sub(Duration::from_secs(2))
                    .expect("test clock must permit a two-second offset"),
            ),
            ..Default::default()
        };
        let bab_deadline = now
            .checked_sub(Duration::from_secs(1))
            .expect("test clock must permit a one-second offset");

        let rebound =
            rebind_missing_or_expired_alpha_deadline_for_bab(&inherited, now, bab_deadline);

        assert_eq!(rebound.deadline, inherited.deadline);
    }

    #[test]
    fn phase_deadline_is_quarter_of_short_global_remainder() {
        let now = Instant::now();
        let inherited = AlphaCrownConfig {
            deadline: None,
            ..Default::default()
        };

        let deadline = per_disjunct_phase_deadline(&inherited, now, now + Duration::from_secs(40));

        assert_eq!(deadline, now + Duration::from_secs(10));
    }

    #[test]
    fn phase_deadline_is_capped_at_fifteen_seconds() {
        let now = Instant::now();
        let inherited = AlphaCrownConfig {
            deadline: None,
            ..Default::default()
        };

        let deadline = per_disjunct_phase_deadline(&inherited, now, now + Duration::from_mins(2));

        assert_eq!(deadline, now + Duration::from_secs(15));
    }

    #[test]
    fn phase_deadline_preserves_live_earlier_warmup_deadline() {
        let now = Instant::now();
        let warmup_deadline = now + Duration::from_secs(3);
        let inherited = AlphaCrownConfig {
            deadline: Some(warmup_deadline),
            ..Default::default()
        };

        let deadline = per_disjunct_phase_deadline(&inherited, now, now + Duration::from_mins(2));

        assert_eq!(deadline, warmup_deadline);
    }

    #[test]
    fn scheduler_spends_one_total_iteration_budget_evenly() {
        let now = Instant::now();
        let mut budget = PerDisjunctBudget::new(10, 3, now + Duration::from_secs(12))
            .expect("ten iterations can provide at least two per row");

        let allocations = [
            budget.next(now).unwrap(),
            budget.next(now).unwrap(),
            budget.next(now).unwrap(),
        ];

        assert_eq!(
            allocations.map(|allocation| allocation.iterations),
            [3, 3, 4]
        );
        assert_eq!(
            allocations
                .iter()
                .map(|allocation| allocation.iterations)
                .sum::<usize>(),
            10
        );
        assert!(budget.next(now).is_none());
    }

    #[test]
    fn scheduler_refuses_less_than_two_iterations_per_row() {
        let now = Instant::now();
        assert!(PerDisjunctBudget::new(5, 3, now + Duration::from_secs(12)).is_none());
        assert!(PerDisjunctBudget::new(6, 3, now + Duration::from_secs(12)).is_some());
    }

    #[test]
    fn twenty_iteration_budget_matches_competition_row_counts() {
        let now = Instant::now();
        let deadline = now + Duration::from_secs(15);
        for (rows, expected_first) in [
            (1usize, Some(20usize)),
            (2, Some(10)),
            (9, Some(2)),
            (10, Some(2)),
            (11, None),
            (99, None),
        ] {
            let got = PerDisjunctBudget::new(20, rows, deadline)
                .and_then(|mut budget| budget.next(now))
                .map(|allocation| allocation.iterations);
            assert_eq!(got, expected_first, "unexpected allocation for {rows} rows");
        }
    }

    #[test]
    fn scheduler_reclaims_early_finished_wall_time_for_remaining_rows() {
        let now = Instant::now();
        let phase_deadline = now + Duration::from_secs(12);
        let mut budget =
            PerDisjunctBudget::new(6, 3, phase_deadline).expect("valid iteration budget");

        let first = budget.next(now).unwrap();
        assert_eq!(first.deadline, now + Duration::from_secs(4));

        // The first row finished after one second, leaving its unused three
        // seconds in the phase pool. The next two rows split all eleven seconds.
        let second_now = now + Duration::from_secs(1);
        let second = budget.next(second_now).unwrap();
        assert_eq!(second.deadline, second_now + Duration::from_millis(5_500));

        let third_now = now + Duration::from_secs(2);
        let third = budget.next(third_now).unwrap();
        assert_eq!(third.deadline, phase_deadline);
    }
}
