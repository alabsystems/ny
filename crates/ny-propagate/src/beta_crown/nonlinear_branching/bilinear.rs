// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use ny_core::Result;
use ny_tensor::BoundedTensor;

use super::{BranchingDecision, NonlinearBranching};
use crate::beta_crown::branching::LayerRef;
use crate::contiguous_flat_slice;
use crate::network::GraphNode;

impl NonlinearBranching {
    /// Score neurons of a BilinearCrown input for split candidacy.
    ///
    /// For BilinearCrown z = Q @ K^T, splitting refines Q or K input intervals.
    /// Tighter input intervals produce tighter McCormick envelopes automatically.
    /// The McCormick relaxation error for z[i,j] = sum_l q[i,l]*k[l,j] is bounded
    /// by (q_u - q_l) * (k_u - k_l) / 4 per element.
    ///
    /// Score = width * partner_avg_width, so splits are prioritized where both
    /// inputs have wide intervals (high McCormick error product).
    ///
    /// Reference: auto_LiRPA `BoundMatMul.splittable = True` (linear.py:948),
    /// `get_split_nodes` (beta_crown.py:62-79) scores both inputs as candidates.
    /// McCormick (1976), "Computability of global solutions to factorable nonconvex programs."
    pub(super) fn score_bilinear_input_neurons(
        &self,
        bilinear_node_name: &str,
        input_idx: usize,
        bounds: &BoundedTensor,
        partner_avg_width: f32,
    ) -> Result<Vec<BranchingDecision>> {
        let mut decisions = Vec::new();

        let lower = contiguous_flat_slice(bounds.lower());
        let upper = contiguous_flat_slice(bounds.upper());

        // Use partner_avg_width as a multiplier; fall back to 1.0 if zero/NaN
        // so within-input ranking is preserved when partner is unavailable.
        let multiplier = if partner_avg_width.is_finite() && partner_avg_width > 0.0 {
            partner_avg_width
        } else {
            1.0
        };

        for (idx, (&l, &u)) in lower.iter().zip(upper.iter()).enumerate() {
            if !l.is_finite() || !u.is_finite() {
                continue;
            }

            let width = u - l;
            if width < self.config.min_branch_width {
                continue;
            }

            // Uniform midpoint branching for bilinear inputs.
            // auto_LiRPA skips optimized branching points for MatMul (bp_opt.py:231-234)
            // because MatMul is not element-wise.
            let n = self.config.num_branches;
            let points: Vec<f32> = (1..n).map(|i| l + width * (i as f32 / n as f32)).collect();
            if points.is_empty() {
                continue;
            }

            // McCormick product score: width * partner_avg_width.
            // The McCormick relaxation error is proportional to the product of both
            // inputs' widths. Multiplying by the partner's average width ensures that
            // when comparing Q vs K split candidates, we favor splitting the input
            // whose partner has wider intervals (higher error product).
            let score = width * multiplier;

            decisions.push(BranchingDecision {
                layer: LayerRef::Name(bilinear_node_name.to_string()),
                neuron_idx: idx,
                points,
                score,
                original_bounds: (l, u),
                input_index: Some(input_idx),
                norm_inv_rms: None,
            });
        }

        Ok(decisions)
    }

    /// Score neurons of a MulBinary node's inputs for split candidacy.
    ///
    /// MulBinary computes the element-wise product z = x · y (e.g. ml4acopf
    /// power-flow x·y, SwiGLU gating). Like BilinearCrown it is McCormick-relaxed,
    /// but the product is element-wise rather than a matmul, so the relaxation
    /// error for element z[i] = x[i]·y[i] is bounded by the PER-ELEMENT gap
    /// (ux[i] − lx[i]) · (uy[i] − ly[i]) / 4.
    ///
    /// We score each (input, element) candidate by `width(input) * partner_avg_width`
    /// — the same sound McCormick-product proxy as `score_bilinear_input_neurons`.
    /// The wider input axis scores higher, so the global candidate sort naturally
    /// splits the axis that most reduces the dominant factor of the envelope gap.
    /// Both inputs (x and y) are emitted with their `input_index`, so
    /// `with_general_split` tightens the correct input node (mirroring BilinearCrown
    /// Q/K splitting).
    ///
    /// SOUNDNESS: this only ranks and selects which sound sub-box to bound next; the
    /// child's bound is still produced by the directed-rounded McCormick
    /// concretization path. The McCormick gap is used purely as a ranking signal and
    /// never folded into any bound.
    ///
    /// Reference: McCormick (1976); auto_LiRPA `BoundMul` / `get_split_nodes`.
    pub(super) fn score_mul_binary_input_neurons(
        &self,
        mul_node_name: &str,
        node: &GraphNode,
        node_bounds: &HashMap<String, BoundedTensor>,
    ) -> Result<Vec<BranchingDecision>> {
        // MulBinary always has exactly two inputs (x, y). Resolve each input's
        // bounds; missing bounds (e.g. a constant operand folded elsewhere) simply
        // drop that input as a split target.
        let input_bounds: Vec<Option<&BoundedTensor>> = node
            .inputs
            .iter()
            .map(|name| node_bounds.get(name.as_str()))
            .collect();

        // Per-input average widths, used as the partner factor in the gap proxy.
        let avg_widths: Vec<f32> = input_bounds
            .iter()
            .map(|b| b.map_or(0.0, Self::compute_avg_width))
            .collect();

        let mut decisions = Vec::new();
        for (input_idx, maybe_bounds) in input_bounds.iter().enumerate() {
            let Some(bounds) = maybe_bounds else {
                continue;
            };
            // Partner is the other operand (MulBinary is strictly binary).
            let partner_idx = 1 - input_idx;
            let partner_avg_width = avg_widths.get(partner_idx).copied().unwrap_or(1.0);
            let mut input_decisions = self.score_bilinear_input_neurons(
                mul_node_name,
                input_idx,
                bounds,
                partner_avg_width,
            )?;
            decisions.append(&mut input_decisions);
        }

        Ok(decisions)
    }
}
