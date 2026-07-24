// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cutting plane pool for sequential (non-graph) GCP-CROWN.
//!
//! Manages a collection of linear cutting planes derived from verified BaB subdomains.
//! When a domain is proved safe, the proof certificates can be reused as cuts that
//! tighten bounds for sibling domains. The pool handles capacity management via
//! configurable eviction policies (FIFO, utility-weighted scoring, stale-guard).
//!
//! Key type: [`CutPool`] — add cuts via `add_cut`, apply to bound computations via
//! iteration over `cuts`. Pool statistics track generation, eviction, and staleness.
//!
//! # Submodules
//! - [`arelu`] — Arelu state builder for cut integration
//! - [`merge`] — BICCOS-style cut merging and deduplication
//! - [`proactive`] — Proactive cut generation (BICCOS-lite)

mod arelu;
mod merge;
mod proactive;
#[cfg(test)]
mod tests;

use std::sync::atomic::{AtomicUsize, Ordering};

use ny_core::nan_propagating_max;

use super::{CutKind, CuttingPlane};
use crate::beta_crown::branching::SplitHistory;
use crate::beta_crown::config::{BetaCrownConfig, CutEvictionPolicy, CutScoreWeights};

/// Pool of cutting planes for GCP-CROWN.
///
/// Manages the collection of cuts derived from verified subdomains during B&B.
/// Cuts are added when domains are verified and applied to all subsequent
/// bound computations.
#[derive(Debug)]
pub struct CutPool {
    /// Active cutting planes.
    pub cuts: Vec<CuttingPlane>,
    /// Maximum number of cuts to retain.
    pub max_cuts: usize,
    /// Minimum depth required to generate cuts.
    pub min_cut_depth: usize,
    /// Total cuts generated (for statistics).
    pub total_generated: usize,
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
    /// Live cut count by kind [Verified, NearMiss, Proactive].
    /// Maintained incrementally to avoid O(n) count_kind scans.
    cuts_live_by_kind: [usize; 3],
    /// Total cuts evicted.
    pub cuts_evicted_total: usize,
    /// Cuts evicted by kind (Verified, NearMiss, Proactive).
    pub cuts_evicted_by_kind: [usize; 3],
    /// Cuts evicted due to stale guards.
    pub cuts_evicted_stale: usize,
    /// Cuts evicted via scoring.
    pub cuts_evicted_score: usize,
}

impl Clone for CutPool {
    fn clone(&self) -> Self {
        Self {
            cuts: self.cuts.clone(),
            max_cuts: self.max_cuts,
            min_cut_depth: self.min_cut_depth,
            total_generated: self.total_generated,
            iter_counter: AtomicUsize::new(self.iter_counter.load(Ordering::Relaxed)),
            eviction_policy: self.eviction_policy,
            cut_stale_iters: self.cut_stale_iters,
            cut_hard_stale_iters: self.cut_hard_stale_iters,
            cut_lambda_min: self.cut_lambda_min,
            cut_proactive_fraction: self.cut_proactive_fraction,
            cut_score_weights: self.cut_score_weights.clone(),
            cuts_live_by_kind: self.cuts_live_by_kind,
            cuts_evicted_total: self.cuts_evicted_total,
            cuts_evicted_by_kind: self.cuts_evicted_by_kind,
            cuts_evicted_stale: self.cuts_evicted_stale,
            cuts_evicted_score: self.cuts_evicted_score,
        }
    }
}

impl Default for CutPool {
    fn default() -> Self {
        Self::new(0)
    }
}

impl CutPool {
    /// Create a new cut pool with specified capacity.
    pub fn new(max_cuts: usize) -> Self {
        Self {
            cuts: Vec::with_capacity(max_cuts),
            max_cuts,
            min_cut_depth: 2,
            total_generated: 0,
            iter_counter: AtomicUsize::new(0),
            eviction_policy: CutEvictionPolicy::default(),
            cut_stale_iters: 200,
            cut_hard_stale_iters: 1000,
            cut_lambda_min: 1e-3,
            cut_proactive_fraction: 0.2,
            cut_score_weights: CutScoreWeights::default(),
            cuts_live_by_kind: [0; 3],
            cuts_evicted_total: 0,
            cuts_evicted_by_kind: [0; 3],
            cuts_evicted_stale: 0,
            cuts_evicted_score: 0,
        }
    }

    /// Create a new cut pool from a verifier configuration.
    pub fn from_config(config: &BetaCrownConfig) -> Self {
        Self {
            cuts: Vec::with_capacity(config.max_cuts),
            max_cuts: config.max_cuts,
            min_cut_depth: config.min_cut_depth,
            total_generated: 0,
            iter_counter: AtomicUsize::new(0),
            eviction_policy: config.cut_eviction_policy,
            cut_stale_iters: config.cut_stale_iters,
            cut_hard_stale_iters: config.cut_hard_stale_iters,
            cut_lambda_min: config.cut_lambda_min,
            cut_proactive_fraction: config.cut_proactive_fraction,
            cut_score_weights: config.cut_score_weights.clone(),
            cuts_live_by_kind: [0; 3],
            cuts_evicted_total: 0,
            cuts_evicted_by_kind: [0; 3],
            cuts_evicted_stale: 0,
            cuts_evicted_score: 0,
        }
    }

    /// Add a cut from a verified domain.
    ///
    /// Returns true if the cut was added, false if pool is full or cut is trivial.
    pub fn add_from_verified_domain(&mut self, history: &SplitHistory) -> ny_core::Result<bool> {
        // Don't add cuts from shallow domains (less likely to be useful)
        if history.depth() < self.min_cut_depth {
            return Ok(false);
        }

        if let Some(cut) = CuttingPlane::from_verified_domain(history)? {
            self.total_generated += 1;
            cut.metadata
                .reset(self.iter_counter.load(Ordering::Relaxed), CutKind::Verified);
            return Ok(self.insert_cut(cut, CutKind::Verified));
        }
        Ok(false)
    }

    /// Add a precomputed cut to the pool.
    pub fn add_cut(&mut self, cut: CuttingPlane) -> bool {
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
    ///
    /// Pre-allocates the result vector to avoid repeated reallocations (#2326).
    pub fn relevant_cuts_for(&self, history: &SplitHistory) -> Vec<&CuttingPlane> {
        let iter = self.bump_iter();
        let mut relevant = Vec::with_capacity(self.cuts.len());
        for cut in &self.cuts {
            if !cut.is_redundant_for(history) {
                cut.metadata.note_used(iter);
                relevant.push(cut);
            }
        }
        relevant
    }

    /// Get mutable references to all cuts for optimization.
    pub fn cuts_mut(&mut self) -> &mut [CuttingPlane] {
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

    /// Sum of all lambda values (for regularization/monitoring).
    pub fn total_lambda(&self) -> f32 {
        self.cuts.iter().map(|c| c.lambda).sum()
    }

    fn bump_iter(&self) -> usize {
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

    fn kind_index(kind: CutKind) -> usize {
        match kind {
            CutKind::Verified => 0,
            CutKind::NearMiss => 1,
            CutKind::Proactive => 2,
        }
    }

    fn count_kind(&self, kind: CutKind) -> usize {
        self.cuts_live_by_kind[Self::kind_index(kind)]
    }

    pub(super) fn rebuild_live_counts(&mut self) {
        self.cuts_live_by_kind = [0; 3];
        for cut in &self.cuts {
            self.cuts_live_by_kind[Self::kind_index(cut.metadata.cut_kind())] += 1;
        }
    }

    fn cut_age(&self, cut: &CuttingPlane, current_iter: usize) -> usize {
        let use_count = cut.metadata.use_count.load(Ordering::Relaxed);
        let base = if use_count == 0 {
            cut.metadata.created_iter.load(Ordering::Relaxed)
        } else {
            cut.metadata.last_used_iter.load(Ordering::Relaxed)
        };
        current_iter.saturating_sub(base)
    }

    pub(super) fn insert_cut(&mut self, cut: CuttingPlane, kind: CutKind) -> bool {
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
            self.cuts_live_by_kind[Self::kind_index(kind)] += 1;
            self.cuts.push(cut);
            return true;
        }
        if self.evict_one(kind) {
            self.cuts_live_by_kind[Self::kind_index(kind)] += 1;
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
        let kind_idx = Self::kind_index(evicted.metadata.cut_kind());
        self.cuts_live_by_kind[kind_idx] -= 1;
        self.cuts_evicted_by_kind[kind_idx] += 1;
        if stale_reason {
            self.cuts_evicted_stale += 1;
        } else {
            self.cuts_evicted_score += 1;
        }
    }

    fn score_cut(&self, cut: &CuttingPlane, current_iter: usize) -> f32 {
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
