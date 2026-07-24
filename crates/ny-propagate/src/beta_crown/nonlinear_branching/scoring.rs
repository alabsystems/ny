// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::Result;
use ny_tensor::BoundedTensor;

use super::{
    BranchingDecision, BranchingPointMethod, NonlinearBranching, NonlinearHeuristicMethod,
};
use crate::beta_crown::branching::LayerRef;
use crate::contiguous_flat_slice;
use crate::layers::Layer;

impl NonlinearBranching {
    /// Score neurons in a node and return branching decisions.
    pub(super) fn score_neurons(
        &self,
        node_name: &str,
        layer: &Layer,
        bounds: &BoundedTensor,
    ) -> Result<Vec<BranchingDecision>> {
        let mut decisions = Vec::new();

        let lower = contiguous_flat_slice(bounds.lower());
        let upper = contiguous_flat_slice(bounds.upper());

        for (idx, (&l, &u)) in lower.iter().zip(upper.iter()).enumerate() {
            // Skip neurons with non-finite bounds (NaN or Inf from numerical instability).
            // NaN bounds produce NaN branching points and NaN scores, which propagate
            // through the BaB tree creating unsound child domains. See #2882.
            if !l.is_finite() || !u.is_finite() {
                continue;
            }

            // Skip neurons with bounds too tight to branch
            let width = u - l;
            if width < self.config.min_branch_width {
                continue;
            }

            // Special case: ReLU and Sign only branch if unstable (crosses 0).
            // Both are zero-threshold activations with identical split semantics.
            if matches!(layer, Layer::ReLU(_) | Layer::Sign(_)) && (l >= 0.0 || u <= 0.0) {
                continue;
            }

            // Compute branching points
            let points = self.branching_points(l, u, layer);
            if points.is_empty() {
                continue;
            }

            let score = self.compute_score(l, u, layer, &points);
            decisions.push(BranchingDecision {
                layer: LayerRef::Name(node_name.to_string()),
                neuron_idx: idx,
                points,
                score,
                original_bounds: (l, u),
                input_index: None,
                norm_inv_rms: None,
            });
        }

        Ok(decisions)
    }

    /// Compute average interval width across all elements of a BoundedTensor.
    pub(super) fn compute_avg_width(bounds: &BoundedTensor) -> f32 {
        let n = bounds.lower().len();
        if n == 0 {
            return 0.0;
        }
        let sum: f32 = bounds
            .lower()
            .iter()
            .zip(bounds.upper().iter())
            .map(|(&l, &u)| {
                let w = u - l;
                if w.is_finite() {
                    w
                } else {
                    0.0
                }
            })
            .sum();
        sum / n as f32
    }

    /// Branching points for a neuron.
    pub(super) fn branching_points(&self, lower: f32, upper: f32, layer: &Layer) -> Vec<f32> {
        // Special case: ReLU and Sign always branch at 0.
        // Both are zero-threshold activations; stable neurons produce no branching points.
        if matches!(layer, Layer::ReLU(_) | Layer::Sign(_)) {
            if lower < 0.0 && upper > 0.0 {
                return vec![0.0];
            }
            return vec![];
        }

        match self.config.point_method {
            BranchingPointMethod::Uniform => {
                let n = self.config.num_branches;
                (1..n)
                    .map(|i| lower + (upper - lower) * (i as f32 / n as f32))
                    .collect()
            }
        }
    }

    /// Compute heuristic score for branching a neuron.
    pub(super) fn compute_score(
        &self,
        lower: f32,
        upper: f32,
        layer: &Layer,
        _points: &[f32],
    ) -> f32 {
        match self.config.method {
            NonlinearHeuristicMethod::BoundWidth => upper - lower,
            NonlinearHeuristicMethod::Bbps => {
                let width = upper - lower;

                match layer {
                    Layer::ReLU(_) | Layer::Sign(_) => {
                        // Sign uses identical BBPS scoring as ReLU: intervals
                        // where zero is near one edge score highest, because
                        // splitting there makes one child very narrow.
                        let dist_to_zero = lower.abs().min(upper.abs());
                        width * (1.0 - dist_to_zero / width.max(1e-6))
                    }
                    Layer::GELU(_) | Layer::SiLU(_) => {
                        let center = f32::midpoint(lower, upper);
                        width * (1.0 + (-center.abs()).exp())
                    }
                    Layer::Sigmoid(_) | Layer::Tanh(_) => {
                        let center = f32::midpoint(lower, upper);
                        width * (1.0 + (-center.abs() / 2.0).exp())
                    }
                    Layer::Softplus(_) => {
                        let center = f32::midpoint(lower, upper);
                        width * (1.0 + (-center.abs()).exp() * 0.5)
                    }
                    _ => width,
                }
            }
        }
    }
}
