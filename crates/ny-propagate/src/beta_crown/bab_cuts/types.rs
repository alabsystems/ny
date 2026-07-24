// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use crate::beta_crown::config::CutScoreWeights;

/// A term in a cutting plane constraint.
///
/// Represents a coefficient for a specific neuron's pre-activation value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutTerm {
    /// Layer index of the neuron.
    pub layer_idx: usize,
    /// Neuron index within the layer.
    pub neuron_idx: usize,
    /// Coefficient in the linear constraint.
    /// Positive for active constraints, negative for inactive.
    pub coefficient: f32,
}

/// Source kind for cut generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutKind {
    Verified,
    NearMiss,
    Proactive,
}

impl CutKind {
    pub(crate) fn bonus(self, weights: &CutScoreWeights) -> f32 {
        match self {
            CutKind::Verified => weights.verified_bonus,
            CutKind::NearMiss => weights.near_miss_bonus,
            CutKind::Proactive => weights.proactive_bonus,
        }
    }

    fn to_u32(self) -> u32 {
        match self {
            CutKind::Verified => 0,
            CutKind::NearMiss => 1,
            CutKind::Proactive => 2,
        }
    }

    fn from_u32(v: u32) -> Self {
        match v {
            0 => CutKind::Verified,
            1 => CutKind::NearMiss,
            _ => CutKind::Proactive,
        }
    }
}

/// Metadata for tracking cut freshness and utility.
///
/// Uses atomic types instead of `Cell` so that `GraphCutPool` (which contains
/// `CutMetadata` via `GraphCuttingPlane`) is `Sync` and can be shared as a
/// read-only view in Rayon `par_iter()` closures. Part of #3813.
///
/// All operations use `Relaxed` ordering — these are statistical counters for
/// eviction heuristics, not synchronization points.
#[derive(Debug)]
pub struct CutMetadata {
    pub created_iter: AtomicUsize,
    pub last_used_iter: AtomicUsize,
    pub use_count: AtomicU32,
    last_contribution_bits: AtomicU32,
    avg_contribution_bits: AtomicU32,
    cut_kind_raw: AtomicU32,
}

impl Clone for CutMetadata {
    fn clone(&self) -> Self {
        Self {
            created_iter: AtomicUsize::new(self.created_iter.load(Ordering::Relaxed)),
            last_used_iter: AtomicUsize::new(self.last_used_iter.load(Ordering::Relaxed)),
            use_count: AtomicU32::new(self.use_count.load(Ordering::Relaxed)),
            last_contribution_bits: AtomicU32::new(
                self.last_contribution_bits.load(Ordering::Relaxed),
            ),
            avg_contribution_bits: AtomicU32::new(
                self.avg_contribution_bits.load(Ordering::Relaxed),
            ),
            cut_kind_raw: AtomicU32::new(self.cut_kind_raw.load(Ordering::Relaxed)),
        }
    }
}

impl CutMetadata {
    pub fn new(iter: usize, cut_kind: CutKind) -> Self {
        Self {
            created_iter: AtomicUsize::new(iter),
            last_used_iter: AtomicUsize::new(iter),
            use_count: AtomicU32::new(0),
            last_contribution_bits: AtomicU32::new(0f32.to_bits()),
            avg_contribution_bits: AtomicU32::new(0f32.to_bits()),
            cut_kind_raw: AtomicU32::new(cut_kind.to_u32()),
        }
    }

    pub fn reset(&self, iter: usize, cut_kind: CutKind) {
        self.created_iter.store(iter, Ordering::Relaxed);
        self.last_used_iter.store(iter, Ordering::Relaxed);
        self.use_count.store(0, Ordering::Relaxed);
        self.last_contribution_bits
            .store(0f32.to_bits(), Ordering::Relaxed);
        self.avg_contribution_bits
            .store(0f32.to_bits(), Ordering::Relaxed);
        self.cut_kind_raw
            .store(cut_kind.to_u32(), Ordering::Relaxed);
    }

    pub fn note_used(&self, iter: usize) {
        self.last_used_iter.store(iter, Ordering::Relaxed);
        let prev = self.use_count.load(Ordering::Relaxed);
        self.use_count
            .store(prev.saturating_add(1), Ordering::Relaxed);
    }

    pub fn note_contribution(&self, contribution: f32) {
        let contribution = contribution.abs();
        self.last_contribution_bits
            .store(contribution.to_bits(), Ordering::Relaxed);
        let prev = f32::from_bits(self.avg_contribution_bits.load(Ordering::Relaxed));
        let alpha = 0.2;
        let next = if prev == 0.0 {
            contribution
        } else {
            alpha * contribution + (1.0 - alpha) * prev
        };
        self.avg_contribution_bits
            .store(next.to_bits(), Ordering::Relaxed);
    }

    /// Read the cut kind.
    pub fn cut_kind(&self) -> CutKind {
        CutKind::from_u32(self.cut_kind_raw.load(Ordering::Relaxed))
    }

    /// Read the last contribution value.
    pub fn last_contribution(&self) -> f32 {
        f32::from_bits(self.last_contribution_bits.load(Ordering::Relaxed))
    }

    /// Read the average contribution value.
    pub fn avg_contribution(&self) -> f32 {
        f32::from_bits(self.avg_contribution_bits.load(Ordering::Relaxed))
    }
}

/// A single term in a GraphCuttingPlane.
///
/// Unlike CutTerm which uses layer_idx, this uses node_name to identify
/// ReLU nodes in DAG structures (e.g., ResNets with skip connections).
#[derive(Debug, Clone)]
pub struct GraphCutTerm {
    /// Name of the ReLU node in the graph.
    pub node_name: String,
    /// Neuron index within the ReLU node's output.
    pub neuron_idx: usize,
    /// Coefficient: +1.0 if active constraint, -1.0 if inactive.
    pub coefficient: f32,
}
