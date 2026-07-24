// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cutting plane pool for graph-based GCP-CROWN.
//!
//! Graph-aware variant of [`super::cut_pool::CutPool`] that references neurons by
//! `(node_name, neuron_index)` pairs instead of flat layer indices. Manages cuts
//! derived from verified subdomains during DAG branch-and-bound, with the same
//! eviction policies (FIFO, utility-weighted, stale-guard) as the sequential pool.
//!
//! Key type: [`GraphCutPool`] -- parallel structure to `CutPool` but using
//! [`GraphCuttingPlane`] and [`GraphCutTerm`] for named-node addressing.

mod merging;
mod proactive;

#[cfg(test)]
mod tests;

use std::sync::atomic::{AtomicUsize, Ordering};

use ny_core::nan_propagating_max;

use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::config::{BetaCrownConfig, CutEvictionPolicy, CutScoreWeights};

use super::{CutKind, GraphCuttingPlane};

/// Pool of cutting planes for GraphNetwork GCP-CROWN.
///
/// Manages graph-aware cuts derived from verified subdomains during B&B.
#[derive(Debug)]
pub struct GraphCutPool {
    /// Active cutting planes.
    pub cuts: Vec<GraphCuttingPlane>,
    /// Maximum number of cuts to retain.
    pub max_cuts: usize,
    /// Total cuts generated (for statistics).
    pub total_generated: usize,
    /// Minimum depth required to generate cuts (default: 2).
    pub min_cut_depth: usize,
    /// Cut pool iteration counter.
    pub iter_counter: AtomicUsize,
    /// Eviction policy for full pools.
    pub eviction_policy: CutEvictionPolicy,
    /// Iteration threshold for stale cuts.
    pub cut_stale_iters: usize,
    /// Iteration threshold for hard-stale cuts.
    pub cut_hard_stale_iters: usize,
    /// Lambda threshold below which stale cuts are evicted.
    pub cut_lambda_min: f32,
    /// Maximum fraction of proactive cuts in the pool.
    pub cut_proactive_fraction: f32,
    /// Scoring weights for utility-weighted eviction.
    pub cut_score_weights: CutScoreWeights,
    /// Total cuts evicted.
    pub cuts_evicted_total: usize,
    /// Cuts evicted by kind (Verified, NearMiss, Proactive).
    pub cuts_evicted_by_kind: [usize; 3],
    /// Cuts evicted due to stale guards.
    pub cuts_evicted_stale: usize,
    /// Cuts evicted via scoring.
    pub cuts_evicted_score: usize,
}

impl Clone for GraphCutPool {
    fn clone(&self) -> Self {
        Self {
            cuts: self.cuts.clone(),
            max_cuts: self.max_cuts,
            total_generated: self.total_generated,
            min_cut_depth: self.min_cut_depth,
            iter_counter: AtomicUsize::new(self.iter_counter.load(Ordering::Relaxed)),
            eviction_policy: self.eviction_policy,
            cut_stale_iters: self.cut_stale_iters,
            cut_hard_stale_iters: self.cut_hard_stale_iters,
            cut_lambda_min: self.cut_lambda_min,
            cut_proactive_fraction: self.cut_proactive_fraction,
            cut_score_weights: self.cut_score_weights.clone(),
            cuts_evicted_total: self.cuts_evicted_total,
            cuts_evicted_by_kind: self.cuts_evicted_by_kind,
            cuts_evicted_stale: self.cuts_evicted_stale,
            cuts_evicted_score: self.cuts_evicted_score,
        }
    }
}

impl Default for GraphCutPool {
    fn default() -> Self {
        Self::new(0)
    }
}

impl GraphCutPool {
    /// Create a new graph cut pool with specified capacity.
    pub fn new(max_cuts: usize) -> Self {
        Self {
            cuts: Vec::with_capacity(max_cuts),
            max_cuts,
            total_generated: 0,
            min_cut_depth: 2,
            iter_counter: AtomicUsize::new(0),
            eviction_policy: CutEvictionPolicy::default(),
            cut_stale_iters: 200,
            cut_hard_stale_iters: 1000,
            cut_lambda_min: 1e-3,
            cut_proactive_fraction: 0.2,
            cut_score_weights: CutScoreWeights::default(),
            cuts_evicted_total: 0,
            cuts_evicted_by_kind: [0; 3],
            cuts_evicted_stale: 0,
            cuts_evicted_score: 0,
        }
    }

    /// Create with custom minimum depth.
    pub fn with_min_depth(max_cuts: usize, min_depth: usize) -> Self {
        Self {
            cuts: Vec::with_capacity(max_cuts),
            max_cuts,
            total_generated: 0,
            min_cut_depth: min_depth,
            iter_counter: AtomicUsize::new(0),
            eviction_policy: CutEvictionPolicy::default(),
            cut_stale_iters: 200,
            cut_hard_stale_iters: 1000,
            cut_lambda_min: 1e-3,
            cut_proactive_fraction: 0.2,
            cut_score_weights: CutScoreWeights::default(),
            cuts_evicted_total: 0,
            cuts_evicted_by_kind: [0; 3],
            cuts_evicted_stale: 0,
            cuts_evicted_score: 0,
        }
    }

    /// Create a new graph cut pool from a verifier configuration.
    pub fn from_config(config: &BetaCrownConfig) -> Self {
        Self {
            cuts: Vec::with_capacity(config.max_cuts),
            max_cuts: config.max_cuts,
            total_generated: 0,
            min_cut_depth: config.min_cut_depth,
            iter_counter: AtomicUsize::new(0),
            eviction_policy: config.cut_eviction_policy,
            cut_stale_iters: config.cut_stale_iters,
            cut_hard_stale_iters: config.cut_hard_stale_iters,
            cut_lambda_min: config.cut_lambda_min,
            cut_proactive_fraction: config.cut_proactive_fraction,
            cut_score_weights: config.cut_score_weights.clone(),
            cuts_evicted_total: 0,
            cuts_evicted_by_kind: [0; 3],
            cuts_evicted_stale: 0,
            cuts_evicted_score: 0,
        }
    }

    /// Add a cut from a verified domain.
    ///
    /// Returns true if the cut was added, false if pool is full or cut is trivial.
    pub fn add_from_verified_domain(
        &mut self,
        history: &GraphSplitHistory,
    ) -> ny_core::Result<bool> {
        // Don't add cuts from shallow domains (less likely to be useful)
        if history.depth() < self.min_cut_depth {
            return Ok(false);
        }

        if let Some(cut) = GraphCuttingPlane::from_verified_domain(history)? {
            self.total_generated += 1;
            cut.metadata
                .reset(self.iter_counter.load(Ordering::Relaxed), CutKind::Verified);
            return Ok(self.insert_cut(cut, CutKind::Verified));
        }
        Ok(false)
    }

    /// Add a cut from a near-miss domain (close to verification but not verified).
    ///
    /// Near-miss cuts are weaker than verified cuts because the domain didn't
    /// actually verify. However, they can still be useful for pruning similar
    /// regions in the search space.
    ///
    /// Returns true if the cut was added, false if pool is full or cut is trivial.
    pub fn add_from_near_miss_domain(
        &mut self,
        history: &GraphSplitHistory,
        lower_bound: f32,
        threshold: f32,
        margin: f32,
    ) -> ny_core::Result<bool> {
        // Don't add cuts from shallow domains
        if history.depth() < self.min_cut_depth {
            return Ok(false);
        }

        // Check if it's a near-miss (close to threshold but not verified)
        let effective_margin = if threshold.abs() < 1e-6 {
            margin // Use absolute margin if threshold is ~0
        } else {
            threshold.abs() * margin // Use relative margin
        };

        // Near-miss: lower_bound is within margin of threshold
        // For verify lower > threshold: lb should be close to threshold but < threshold
        let gap = threshold - lower_bound;
        if gap <= 0.0 || gap > effective_margin {
            return Ok(false); // Not a near-miss (either verified or too far)
        }

        if let Some(cut) = GraphCuttingPlane::from_verified_domain(history)? {
            self.total_generated += 1;
            cut.metadata
                .reset(self.iter_counter.load(Ordering::Relaxed), CutKind::NearMiss);
            return Ok(self.insert_cut(cut, CutKind::NearMiss));
        }
        Ok(false)
    }

    /// Add a precomputed graph cut to the pool.
    pub fn add_cut(&mut self, cut: GraphCuttingPlane) -> bool {
        if cut.terms.is_empty() {
            return false;
        }
        let kind = cut.metadata.cut_kind();
        self.total_generated += 1;
        cut.metadata
            .reset(self.iter_counter.load(Ordering::Relaxed), kind);
        self.insert_cut(cut, kind)
    }

    /// Get cuts that are relevant for a domain (not redundant).
    pub fn relevant_cuts_for(&self, history: &GraphSplitHistory) -> Vec<&GraphCuttingPlane> {
        let iter = self.bump_iter();
        let mut relevant = Vec::new();
        for cut in &self.cuts {
            if !cut.is_redundant_for(history) {
                cut.metadata.note_used(iter);
                relevant.push(cut);
            }
        }
        relevant
    }

    /// Get mutable references to all cuts for optimization.
    pub fn cuts_mut(&mut self) -> &mut [GraphCuttingPlane] {
        &mut self.cuts
    }

    /// Reset all lambda gradients.
    pub fn zero_grad(&mut self) {
        for cut in &mut self.cuts {
            cut.zero_grad();
        }
    }

    /// Number of active cuts.
    pub fn len(&self) -> usize {
        self.cuts.len()
    }

    /// Check if pool is empty.
    pub fn is_empty(&self) -> bool {
        self.cuts.is_empty()
    }

    /// Sum of all lambda values.
    pub fn total_lambda(&self) -> f32 {
        self.cuts.iter().map(|c| c.lambda).sum()
    }

    // ── Internal helpers ─────────────────────────────────────────────

    pub(super) fn bump_iter(&self) -> usize {
        let next = self.iter_counter.load(Ordering::Relaxed).saturating_add(1);
        self.iter_counter.store(next, Ordering::Relaxed);
        next
    }

    fn proactive_limit(&self) -> usize {
        if self.max_cuts == 0 {
            return 0;
        }
        let fraction = self.cut_proactive_fraction.clamp(0.0, 1.0);
        // NaN passes through clamp unchanged; treat as zero.
        if fraction.is_nan() || fraction <= 0.0 {
            return 0;
        }
        // SAFETY: fraction is finite and in (0.0, 1.0], max_cuts > 0,
        // so product is finite and non-negative.
        let limit = (fraction * (self.max_cuts as f32)).ceil() as usize;
        limit.max(1).min(self.max_cuts)
    }

    fn count_kind(&self, kind: CutKind) -> usize {
        self.cuts
            .iter()
            .filter(|cut| cut.metadata.cut_kind() == kind)
            .count()
    }

    fn cut_age(&self, cut: &GraphCuttingPlane, current_iter: usize) -> usize {
        let use_count = cut.metadata.use_count.load(Ordering::Relaxed);
        let base = if use_count == 0 {
            cut.metadata.created_iter.load(Ordering::Relaxed)
        } else {
            cut.metadata.last_used_iter.load(Ordering::Relaxed)
        };
        current_iter.saturating_sub(base)
    }

    pub(super) fn insert_cut(&mut self, cut: GraphCuttingPlane, kind: CutKind) -> bool {
        if self.max_cuts == 0 {
            return false;
        }
        let proactive_limit = self.proactive_limit();
        if kind == CutKind::Proactive && proactive_limit == 0 {
            return false;
        }
        if self.cuts.len() < self.max_cuts {
            if kind == CutKind::Proactive && self.count_kind(CutKind::Proactive) >= proactive_limit
            {
                return false;
            }
            self.cuts.push(cut);
            return true;
        }
        if self.evict_one(kind) {
            self.cuts.push(cut);
            return true;
        }
        false
    }

    fn evict_one(&mut self, incoming_kind: CutKind) -> bool {
        if self.cuts.is_empty() {
            return false;
        }

        let current_iter = self.iter_counter.load(Ordering::Relaxed);
        let proactive_limit = self.proactive_limit();
        let proactive_count = self.count_kind(CutKind::Proactive);
        let verified_count = self.count_kind(CutKind::Verified);

        let mut candidates: Vec<usize> = (0..self.cuts.len()).collect();
        if incoming_kind == CutKind::Proactive && proactive_count >= proactive_limit {
            candidates.retain(|&idx| self.cuts[idx].metadata.cut_kind() == CutKind::Proactive);
            if candidates.is_empty() {
                return false;
            }
        } else if proactive_count > proactive_limit {
            candidates.retain(|&idx| self.cuts[idx].metadata.cut_kind() == CutKind::Proactive);
            if candidates.is_empty() {
                candidates = (0..self.cuts.len()).collect();
            }
        }

        let mut hard_stale: Vec<usize> = candidates
            .iter()
            .cloned()
            .filter(|&idx| {
                let cut = &self.cuts[idx];
                let age = self.cut_age(cut, current_iter);
                // NaN lambda is treated as zero-magnitude (eligible for eviction)
                // because NaN.abs() < threshold is always false in IEEE 754.
                age > self.cut_hard_stale_iters
                    && (cut.lambda.is_nan() || cut.lambda.abs() < self.cut_lambda_min)
            })
            .collect();
        if verified_count == 1 {
            hard_stale.retain(|&idx| self.cuts[idx].metadata.cut_kind() != CutKind::Verified);
        }
        if !hard_stale.is_empty() {
            return self.evict_by_score(&hard_stale, true);
        }

        let stale_candidates: Vec<usize> = candidates
            .iter()
            .cloned()
            .filter(|&idx| {
                let cut = &self.cuts[idx];
                let age = self.cut_age(cut, current_iter);
                let use_count = cut.metadata.use_count.load(Ordering::Relaxed);
                age > self.cut_stale_iters
                    && use_count > 0
                    && (cut.lambda.is_nan() || cut.lambda.abs() < self.cut_lambda_min)
            })
            .collect();
        if !stale_candidates.is_empty() {
            return self.evict_by_score(&stale_candidates, true);
        }

        match self.eviction_policy {
            CutEvictionPolicy::Fifo => {
                let mut oldest_idx = candidates[0];
                let mut oldest_iter = self.cuts[oldest_idx]
                    .metadata
                    .created_iter
                    .load(Ordering::Relaxed);
                for &idx in &candidates[1..] {
                    let created = self.cuts[idx].metadata.created_iter.load(Ordering::Relaxed);
                    if created < oldest_iter {
                        oldest_iter = created;
                        oldest_idx = idx;
                    }
                }
                self.evict_index(oldest_idx, false);
                true
            }
            CutEvictionPolicy::UtilityWeighted => self.evict_by_score(&candidates, false),
        }
    }

    fn evict_by_score(&mut self, candidates: &[usize], stale_reason: bool) -> bool {
        if candidates.is_empty() {
            return false;
        }
        let current_iter = self.iter_counter.load(Ordering::Relaxed);
        let mut best_idx = candidates[0];
        let mut best_score = self.score_cut(&self.cuts[best_idx], current_iter);
        for &idx in &candidates[1..] {
            let score = self.score_cut(&self.cuts[idx], current_iter);
            if score < best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        self.evict_index(best_idx, stale_reason);
        true
    }

    fn evict_index(&mut self, index: usize, stale_reason: bool) {
        let evicted = self.cuts.swap_remove(index);
        self.cuts_evicted_total += 1;
        let kind_idx = match evicted.metadata.cut_kind() {
            CutKind::Verified => 0,
            CutKind::NearMiss => 1,
            CutKind::Proactive => 2,
        };
        self.cuts_evicted_by_kind[kind_idx] += 1;
        if stale_reason {
            self.cuts_evicted_stale += 1;
        } else {
            self.cuts_evicted_score += 1;
        }
    }

    fn score_cut(&self, cut: &GraphCuttingPlane, current_iter: usize) -> f32 {
        let weights = &self.cut_score_weights;
        // NaN-safe: propagate NaN instead of silently clamping to 0.0 (#2643)
        let lambda_cap = nan_propagating_max(weights.lambda_cap, 0.0);
        let contrib_cap = nan_propagating_max(weights.contrib_cap, 0.0);
        // NaN guard: clamp() propagates NaN; treat NaN lambda as 0.0 (#2598)
        let lambda = if cut.lambda.is_finite() {
            cut.lambda.clamp(0.0, lambda_cap)
        } else {
            0.0
        };
        let use_count = cut.metadata.use_count.load(Ordering::Relaxed) as f32;
        let avg = cut.metadata.avg_contribution();
        let avg = if avg.is_finite() { avg } else { 0.0 };
        let avg = avg.clamp(0.0, contrib_cap);
        let depth = (cut.source_depth as f32).ln_1p();
        let age = self.cut_age(cut, current_iter) as f32;
        let tau = weights.tau_iters.max(1.0);
        let recent = (-age / tau).exp();
        let usage = use_count.ln_1p();
        let kind_bonus = cut.metadata.cut_kind().bonus(weights);

        weights.w_lambda * lambda
            + weights.w_recent * recent
            + weights.w_usage * usage
            + weights.w_contrib * avg
            + weights.w_depth * depth
            + kind_bonus
    }
}
