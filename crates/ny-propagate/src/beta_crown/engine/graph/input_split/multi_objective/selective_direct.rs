// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pure scheduling policy for selective direct-conic CROWN escalation.
//!
//! The proof-producing caller owns all objective provenance and bound
//! evaluation. This module only decides which already-authenticated,
//! unresolved domains to try and how to divide those attempts into a bounded
//! live batch. A scheduling decision can therefore change proof search cost,
//! never proof authority.

use std::time::{Duration, Instant};

/// Hard upper bound on simultaneously live direct-CROWN domain rows.
///
/// The selective pass is additive to a just-completed source-row rebound. Keep
/// this deliberately small: selected attempts may span multiple chunks, but a
/// single dispatch can never retain more than eight extra one-row carriers.
pub(super) const SELECTIVE_DIRECT_MICROBATCH_CAP: usize = 8;
/// Spend one optional direct-row attempt per eight eligible source domains.
pub(super) const SELECTIVE_DIRECT_STRIDE: usize = 8;
/// Maximum cumulative BaB wall-time share spent on optional direct CROWN.
pub(super) const SELECTIVE_DIRECT_BAB_FRACTION: f32 = 0.08;
/// Maximum BaB wall-time share spent on the root-only direct probe.
pub(super) const SELECTIVE_DIRECT_ROOT_BAB_FRACTION: f32 = 0.01;

/// Derive a phase-local call deadline from a cumulative optional budget.
///
/// `elapsed` counts only prior optional calls; ordinary source BaB work does
/// not spend this pool. `per_call_cap` lets the root probe use a smaller slice.
/// The returned deadline never exceeds the instance-wide deadline.
pub(super) fn call_deadline(
    now: Instant,
    global_deadline: Option<Instant>,
    total_budget: Duration,
    elapsed: Duration,
    per_call_cap: Duration,
) -> Option<Instant> {
    let slice = total_budget.saturating_sub(elapsed).min(per_call_cap);
    if slice.is_zero() {
        return None;
    }
    let local_deadline = now.checked_add(slice).or(global_deadline)?;
    Some(global_deadline.map_or(local_deadline, |global| global.min(local_deadline)))
}

/// One unresolved domain offered to the selective direct-CROWN scheduler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SelectiveDirectCandidate {
    domain_index: usize,
    normalized_affine_gap: f64,
    source_priority: f32,
}

impl SelectiveDirectCandidate {
    /// Construct a candidate from advisory source-affine geometry.
    ///
    /// `affine` is `(gap, lhs_weight, rhs_weight)` for the best already-sound
    /// affine-conic evaluation. The normalized gap is heuristic only; missing
    /// or irregular geometry ranks behind every finite gap and then falls back
    /// to the source queue priority.
    pub(super) fn new(
        domain_index: usize,
        affine: Option<(f64, f32, f32)>,
        source_priority: f32,
    ) -> Self {
        let normalized_affine_gap = affine.map_or(f64::NEG_INFINITY, |(gap, lhs, rhs)| {
            normalize_affine_gap(gap, lhs, rhs)
        });
        let source_priority = if source_priority.is_finite() {
            source_priority
        } else {
            f32::NEG_INFINITY
        };
        Self {
            domain_index,
            normalized_affine_gap,
            source_priority,
        }
    }
    #[cfg(test)]
    fn normalized_affine_gap(self) -> f64 {
        self.normalized_affine_gap
    }
}

/// Normalize a conic gap by the positive multiplier mass.
///
/// Affine closure evaluates both normalized candidates and the `(1, 1)` unit
/// sum. Dividing by `lhs + rhs` prevents an equivalent positive rescaling from
/// receiving a different scheduling rank. Every irregular input maps to
/// negative infinity; this value is advisory and never feeds a verdict.
#[inline]
pub(super) fn normalize_affine_gap(gap: f64, lhs_weight: f32, rhs_weight: f32) -> f64 {
    if !gap.is_finite()
        || !lhs_weight.is_finite()
        || !rhs_weight.is_finite()
        || lhs_weight < 0.0
        || rhs_weight < 0.0
    {
        return f64::NEG_INFINITY;
    }
    let mass = f64::from(lhs_weight) + f64::from(rhs_weight);
    if !mass.is_finite() || mass <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let normalized = gap / mass;
    if normalized.is_finite() {
        normalized
    } else {
        f64::NEG_INFINITY
    }
}

/// Deterministic token-bucket quota across successive queue refreshes.
///
/// Every `stride` unresolved candidates earn one direct-CROWN attempt. Earned
/// attempts are spent only on candidates in the current refresh, ranked by
/// normalized affine gap, then source priority, then ascending domain index.
/// A zero stride is an explicit disabled policy and performs no accounting.
#[derive(Clone, Debug)]
pub(super) struct SelectiveDirectQuota {
    stride: usize,
    candidates_seen: usize,
    selected: usize,
}

impl SelectiveDirectQuota {
    pub(super) fn new(stride: usize) -> Self {
        Self {
            stride,
            candidates_seen: 0,
            selected: 0,
        }
    }

    /// Rank the current refresh and spend every newly available attempt.
    ///
    /// Candidate indices must name distinct entries in the caller's current
    /// domain vector. The returned indices are in scheduling rank order.
    pub(super) fn select(&mut self, candidates: &[SelectiveDirectCandidate]) -> Vec<usize> {
        if self.stride == 0 || candidates.is_empty() {
            return Vec::new();
        }

        self.candidates_seen = self.candidates_seen.saturating_add(candidates.len());
        let earned_total = self.candidates_seen / self.stride;
        let available = earned_total
            .saturating_sub(self.selected)
            .min(candidates.len());
        if available == 0 {
            return Vec::new();
        }

        let mut ranked = candidates.to_vec();
        ranked.sort_unstable_by(|lhs, rhs| {
            rhs.normalized_affine_gap
                .total_cmp(&lhs.normalized_affine_gap)
                .then_with(|| rhs.source_priority.total_cmp(&lhs.source_priority))
                .then_with(|| lhs.domain_index.cmp(&rhs.domain_index))
        });
        ranked.truncate(available);
        self.selected = self.selected.saturating_add(ranked.len());
        ranked
            .into_iter()
            .map(|candidate| candidate.domain_index)
            .collect()
    }

    pub(super) fn candidates_seen(&self) -> usize {
        self.candidates_seen
    }

    pub(super) fn selected(&self) -> usize {
        self.selected
    }
}

/// Split selected domain indices into sequential, hard-bounded live batches.
#[inline]
pub(super) fn selective_direct_chunks(
    selected: &[usize],
    microbatch_cap: usize,
) -> std::slice::Chunks<'_, usize> {
    selected.chunks(microbatch_cap.clamp(1, SELECTIVE_DIRECT_MICROBATCH_CAP))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(index: usize, gap: f64, mass: f32, priority: f32) -> SelectiveDirectCandidate {
        SelectiveDirectCandidate::new(index, Some((gap, mass, 0.0)), priority)
    }

    #[test]
    fn affine_gap_rank_is_invariant_under_positive_rescaling() {
        let unit = normalize_affine_gap(0.2, 1.0, 1.0);
        let normalized = normalize_affine_gap(0.1, 0.5, 0.5);
        assert_eq!(unit.to_bits(), normalized.to_bits());
        assert_eq!(unit, 0.1);
    }

    #[test]
    fn irregular_affine_geometry_ranks_as_negative_infinity() {
        for value in [
            normalize_affine_gap(f64::NAN, 1.0, 1.0),
            normalize_affine_gap(f64::INFINITY, 1.0, 1.0),
            normalize_affine_gap(1.0, f32::NAN, 1.0),
            normalize_affine_gap(1.0, -1.0, 1.0),
            normalize_affine_gap(1.0, 0.0, 0.0),
        ] {
            assert_eq!(value, f64::NEG_INFINITY);
        }
        let missing = SelectiveDirectCandidate::new(0, None, 1.0);
        assert_eq!(missing.normalized_affine_gap(), f64::NEG_INFINITY);
    }

    #[test]
    fn ranking_is_gap_then_priority_then_ascending_index() {
        let candidates = [
            candidate(7, 0.2, 2.0, 100.0), // normalized gap 0.1
            candidate(5, 0.2, 1.0, -10.0), // normalized gap 0.2: first
            candidate(3, 0.1, 1.0, 4.0),   // normalized gap 0.1, high priority
            candidate(1, 0.1, 1.0, 4.0),   // exact tie, lower index first
        ];
        let mut quota = SelectiveDirectQuota::new(1);
        assert_eq!(quota.select(&candidates), vec![5, 7, 1, 3]);
    }

    #[test]
    fn nonfinite_source_priority_loses_a_missing_geometry_tie() {
        let candidates = [
            SelectiveDirectCandidate::new(0, None, f32::NAN),
            SelectiveDirectCandidate::new(1, None, -3.0),
            SelectiveDirectCandidate::new(2, None, f32::INFINITY),
        ];
        let mut quota = SelectiveDirectQuota::new(1);
        assert_eq!(quota.select(&candidates), vec![1, 0, 2]);
    }

    #[test]
    fn quota_earns_one_attempt_per_stride_across_refreshes() {
        let first: Vec<_> = (0..7)
            .map(|index| candidate(index, index as f64, 1.0, 0.0))
            .collect();
        let second = [candidate(8, -1.0, 1.0, 0.0), candidate(7, 1.0, 1.0, 0.0)];
        let mut quota = SelectiveDirectQuota::new(SELECTIVE_DIRECT_STRIDE);

        assert!(quota.select(&first).is_empty());
        assert_eq!(quota.candidates_seen(), 7);
        assert_eq!(quota.selected(), 0);
        assert_eq!(quota.select(&second), vec![7]);
        assert_eq!(quota.candidates_seen(), 9);
        assert_eq!(quota.selected(), 1);
    }

    #[test]
    fn zero_stride_is_disabled_without_accounting() {
        let mut quota = SelectiveDirectQuota::new(0);
        assert!(quota.select(&[candidate(0, 1.0, 1.0, 1.0)]).is_empty());
        assert_eq!(quota.candidates_seen(), 0);
        assert_eq!(quota.selected(), 0);
    }

    #[test]
    fn five_hundred_twelve_candidates_at_stride_eight_select_sixty_four() {
        let candidates: Vec<_> = (0..512)
            .map(|index| candidate(index, index as f64, 1.0, index as f32))
            .collect();
        let mut quota = SelectiveDirectQuota::new(SELECTIVE_DIRECT_STRIDE);
        let selected = quota.select(&candidates);

        assert_eq!(selected.len(), 64);
        assert_eq!(selected[0], 511);
        assert_eq!(selected[63], 448);
        let chunks: Vec<_> =
            selective_direct_chunks(&selected, SELECTIVE_DIRECT_MICROBATCH_CAP).collect();
        assert_eq!(chunks.len(), 8);
        assert!(chunks
            .iter()
            .all(|chunk| !chunk.is_empty() && chunk.len() <= 8));
    }

    #[test]
    fn chunk_policy_is_exact_at_zero_one_eight_and_nine() {
        for (len, expected) in [(0, Vec::new()), (1, vec![1]), (8, vec![8]), (9, vec![8, 1])] {
            let selected: Vec<_> = (0..len).collect();
            let actual: Vec<_> =
                selective_direct_chunks(&selected, SELECTIVE_DIRECT_MICROBATCH_CAP)
                    .map(<[usize]>::len)
                    .collect();
            assert_eq!(actual, expected, "len={len}");
        }
        let selected: Vec<_> = (0..3).collect();
        let singleton_chunks: Vec<_> = selective_direct_chunks(&selected, 1)
            .map(<[usize]>::len)
            .collect();
        assert_eq!(singleton_chunks, vec![1, 1, 1]);
    }

    #[test]
    fn identical_inputs_produce_identical_selection() {
        let candidates = [
            candidate(9, 0.0, 1.0, 2.0),
            candidate(2, 0.0, 1.0, 2.0),
            candidate(5, -1.0, 1.0, 100.0),
            candidate(4, 1.0, 1.0, -100.0),
        ];
        let mut first = SelectiveDirectQuota::new(2);
        let mut second = SelectiveDirectQuota::new(2);
        assert_eq!(first.select(&candidates), second.select(&candidates));
    }

    #[test]
    fn call_deadline_respects_remaining_pool_call_cap_and_global_deadline() {
        let now = Instant::now();
        let total = Duration::from_secs(8);

        let capped = call_deadline(
            now,
            Some(now + Duration::from_secs(20)),
            total,
            Duration::from_secs(2),
            Duration::from_secs(1),
        )
        .expect("one-second root slice remains");
        assert_eq!(capped.duration_since(now), Duration::from_secs(1));

        let remaining = call_deadline(
            now,
            Some(now + Duration::from_secs(20)),
            total,
            Duration::from_secs(2),
            Duration::MAX,
        )
        .expect("six seconds remain");
        assert_eq!(remaining.duration_since(now), Duration::from_secs(6));

        let global = call_deadline(
            now,
            Some(now + Duration::from_millis(500)),
            total,
            Duration::ZERO,
            Duration::from_secs(1),
        )
        .expect("global slice remains");
        assert_eq!(global.duration_since(now), Duration::from_millis(500));

        assert!(call_deadline(now, None, total, total, Duration::MAX).is_none());
    }
}
