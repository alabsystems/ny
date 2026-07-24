// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::Result;

use crate::beta_crown::branching::{LayerRef, NeuronSplit};

/// A branching decision for a single neuron.
#[derive(Debug, Clone)]
pub struct BranchingDecision {
    /// Reference to the layer containing this neuron.
    pub layer: LayerRef,

    /// Neuron index within the layer.
    pub neuron_idx: usize,

    /// Branching points (num_branches - 1 values).
    /// For binary split with point p: branches are [lower, p] and [p, upper].
    pub points: Vec<f32>,

    /// Heuristic score for this candidate (higher = better to branch).
    pub score: f32,

    /// Original bounds of this neuron [lower, upper].
    pub original_bounds: (f32, f32),

    /// For binary operations (e.g., BilinearCrown), which input to split.
    /// `None` = first input (default). `Some(1)` = second input.
    pub input_index: Option<usize>,

    /// GenBaB norm branching marker (#norm-genbab). When `Some((group, lo, hi))`,
    /// this decision splits ONE normalization group (`group`) of a
    /// `Layer::RmsNorm` node's INTERNAL `inv_rms` scalar over the parent range
    /// `[lo, hi]` (not the node's value interval). `to_splits` then emits two
    /// `NeuronSplit`s carrying the lower/upper `inv_rms` halves for that group,
    /// which `with_general_split` turns into [`NormInvRmsConstraint`]s.
    ///
    /// Splitting one group at a time is required for soundness (see
    /// `InvRmsOverride`). `points`/`original_bounds`/`neuron_idx` are unused for
    /// norm splits.
    pub norm_inv_rms: Option<(usize, f32, f32)>,
}

impl BranchingDecision {
    /// Create NeuronSplit instances for all branches from this decision.
    ///
    /// For binary split at point p:
    /// - Branch 0: [original_lower, p] (lower branch)
    /// - Branch 1: [p, original_upper] (upper branch)
    pub fn to_splits(&self) -> Result<Vec<NeuronSplit>> {
        // GenBaB norm branching (#norm-genbab): split ONE group's parent
        // inv_rms range into two children that union-cover that group's range
        // (the other groups stay at full IBP range in both).
        //
        // We split at the GEOMETRIC mean sqrt(lo*hi) rather than the arithmetic
        // midpoint. The IBP inv_rms range is hugely skewed: its upper end is the
        // near-zero-‖x‖ corner (inv_rms → 1/√eps ≈ 316 for eps=1e-5) — a
        // measure-tiny region — while the worst-case objective sits near the
        // lower end (inv_rms ≈ 1 at x = ±1). Arithmetic bisection wastes ~8
        // levels shrinking [1, 316] before the relaxation can beat the fused IBP
        // (empirically the decomposed relaxation only survives once the window
        // narrows below ~0.6); the geometric mean refines the low (load-bearing)
        // region exponentially faster while still soundly covering the high tail
        // in the sibling. Boundary inv_rms == mid lands in BOTH children's closed
        // intervals, so coverage is exact.
        if let Some((group, lo, hi)) = self.norm_inv_rms {
            let mid = if lo > 0.0 && hi > 0.0 && hi > lo {
                ((lo as f64) * (hi as f64)).sqrt() as f32
            } else {
                0.5 * lo + 0.5 * hi
            };
            // Guard against a degenerate split point landing on an endpoint
            // (f32 rounding); fall back to arithmetic midpoint so both children
            // are strictly narrower and BaB makes progress.
            let mid = if mid > lo && mid < hi {
                mid
            } else {
                0.5 * lo + 0.5 * hi
            };
            let lower_child =
                NeuronSplit::norm_inv_rms(self.layer.clone(), group, lo, mid, self.score)?;
            let upper_child =
                NeuronSplit::norm_inv_rms(self.layer.clone(), group, mid, hi, self.score)?;
            return Ok(vec![lower_child, upper_child]);
        }

        let num_branches = self.points.len() + 1;
        let mut splits = Vec::with_capacity(num_branches);

        for i in 0..num_branches {
            let lower = if i == 0 {
                None // Use original lower bound
            } else {
                Some(self.points[i - 1])
            };

            let upper = if i == num_branches - 1 {
                None // Use original upper bound
            } else {
                Some(self.points[i])
            };

            let mut split = NeuronSplit::new(
                self.layer.clone(),
                self.neuron_idx,
                lower,
                upper,
                self.score,
            )?;
            if let Some(idx) = self.input_index {
                split = split.with_input_index(idx);
            }
            splits.push(split);
        }

        Ok(splits)
    }
}
