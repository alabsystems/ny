// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use ny_core::Result;
use ny_tensor::BoundedTensor;

use super::{BranchingDecision, NonlinearBranching};
use crate::layers::Layer;
use crate::network::GraphNetwork;

impl NonlinearBranching {
    /// Get branching decisions for a GraphNetwork domain.
    ///
    /// # Arguments
    /// * `network` - The graph network being verified
    /// * `node_bounds` - Current bounds for each node in the network
    /// * `split_nodes` - Names of nodes eligible for splitting
    ///
    /// # Returns
    /// Sorted list of branching decisions (best candidates first).
    pub fn decisions(
        &self,
        network: &GraphNetwork,
        node_bounds: &HashMap<String, BoundedTensor>,
        split_nodes: &[String],
    ) -> Result<Vec<BranchingDecision>> {
        self.decisions_with_norm_windows(network, node_bounds, split_nodes, None)
    }

    /// [`Self::decisions`] with the domain's already-applied GenBaB norm
    /// `inv_rms` windows (#norm-genbab), so RmsNorm branching computes the
    /// EFFECTIVE (post-constraint) per-group range and converges.
    ///
    /// Without this, the RmsNorm selector would re-derive the same full input-
    /// based `inv_rms` range every iteration (norm splits don't narrow input
    /// bounds) and split the same group forever. Intersecting with the
    /// accumulated history windows shrinks the candidate range each split, so it
    /// eventually drops below `min_branch_width` and branching stops.
    pub fn decisions_with_norm_windows(
        &self,
        network: &GraphNetwork,
        node_bounds: &HashMap<String, BoundedTensor>,
        split_nodes: &[String],
        norm_windows: Option<&HashMap<String, Vec<Option<(f32, f32)>>>>,
    ) -> Result<Vec<BranchingDecision>> {
        let mut candidates = Vec::new();

        for node_name in split_nodes {
            // Get the node and its bounds
            let node = match network.node(node_name) {
                Some(n) => n,
                None => continue,
            };

            // Skip if relu_only and this isn't a ReLU
            if self.config.relu_only && !matches!(node.layer, Layer::ReLU(_)) {
                continue;
            }

            // Check if this layer type is splittable
            if !self.is_splittable(&node.layer) {
                continue;
            }

            // GenBaB norm branching (#norm-genbab): split the RmsNorm node's
            // internal inv_rms scalar. The branchable range is derived from the
            // node's INPUT bounds (inv_rms is a function of the input), not the
            // node's own output value, so it is not a standard neuron split.
            if let Layer::RmsNorm(rms) = &node.layer {
                let applied = norm_windows
                    .and_then(|m| m.get(node_name))
                    .map(|v| v.as_slice());
                if let Some(decision) =
                    self.score_rms_norm_inv_rms(node_name, node, rms.eps, node_bounds, applied)?
                {
                    candidates.push(decision);
                }
                continue;
            }

            // BilinearCrown: score BOTH inputs' neurons as split candidates.
            // Each input (Q, K) of the bilinear node is a separate split target.
            // Reference: auto_LiRPA get_split_nodes (beta_crown.py:62-79) adds both
            // input nodes as split candidates with their respective input_index.
            //
            // McCormick-aware scoring: the relaxation error for z=x*y is bounded by
            // (x_u-x_l)*(y_u-y_l)/4. Score each candidate by width * partner_avg_width
            // so splits are prioritized where both inputs have wide intervals.
            // McCormick (1976), "Computability of global solutions to factorable nonconvex programs."
            if matches!(node.layer, Layer::BilinearCrown(_)) {
                // Compute average width for each input (BilinearCrown always has 2).
                let avg_widths: Vec<f32> = node
                    .inputs
                    .iter()
                    .map(|name| {
                        node_bounds
                            .get(name.as_str())
                            .map_or(0.0, Self::compute_avg_width)
                    })
                    .collect();

                for (input_idx, input_name) in node.inputs.iter().enumerate() {
                    if let Some(bounds) = node_bounds.get(input_name.as_str()) {
                        // Partner's average width: for input 0 (Q), partner is input 1 (K)
                        let partner_idx = 1 - input_idx;
                        let partner_avg_width = avg_widths.get(partner_idx).copied().unwrap_or(1.0);
                        let mut decisions = self.score_bilinear_input_neurons(
                            node_name,
                            input_idx,
                            bounds,
                            partner_avg_width,
                        )?;
                        candidates.append(&mut decisions);
                    }
                }
                continue;
            }

            // MulBinary: element-wise variable×variable product z = x·y. Unlike
            // BilinearCrown (a matmul where every output couples a row of x with a
            // column of y), MulBinary is an element-wise broadcast product, so the
            // McCormick envelope gap is PER-ELEMENT: gap[i] = (ux[i]−lx[i])·(uy[i]−ly[i])/4.
            // We score each (input, element) candidate by the element's own gap and
            // split the WIDER of the two input axes at that element, since halving
            // the wider axis halves the dominant factor of the gap. This is the
            // path ml4acopf's power-flow x·y nodes need to become branchable.
            //
            // SOUNDNESS: scoring/axis selection is search-only — the child bound is
            // still produced by the directed-rounded McCormick concretization.
            if matches!(node.layer, Layer::MulBinary(_)) {
                let mut decisions =
                    self.score_mul_binary_input_neurons(node_name, node, node_bounds)?;
                candidates.append(&mut decisions);
                continue;
            }

            let bounds = match node_bounds.get(node_name) {
                Some(b) => b,
                None => continue,
            };

            // Score each neuron in this node
            let neuron_decisions = self.score_neurons(node_name, &node.layer, bounds)?;
            candidates.extend(neuron_decisions);
        }

        // Sort by score (descending, NaN last — #2995) and return top-k
        candidates.sort_by(|a, b| crate::cmp_utils::nan_last_descending_cmp(&a.score, &b.score));
        candidates.truncate(self.config.num_candidates);
        Ok(candidates)
    }

    /// Check if a layer type is splittable for nonlinear branching.
    ///
    /// Includes elementwise activations (ReLU, GELU, etc.), BilinearCrown
    /// nodes (MatMul with two perturbed inputs), and MulBinary nodes
    /// (element-wise variable×variable products, e.g. ml4acopf power-flow x·y).
    /// Both BilinearCrown and MulBinary are McCormick-relaxed bilinear ops:
    /// splitting either input directly reduces the McCormick envelope gap
    /// (ux−lx)(uy−ly)/4, enabling BaB to close the relaxation error that
    /// frozen root facets cannot. See designs/2026-03-04-286-attention-bilinear-alternative.md
    /// Approach C.
    ///
    /// SOUNDNESS: splittability only adds a candidate to the search; the
    /// resulting child bound is still produced by the directed-rounded
    /// McCormick concretization path. A splittable op never changes a verdict.
    ///
    /// Reference: auto_LiRPA `BoundMatMul.splittable = True` when both inputs are
    /// perturbed (linear.py:948). Source: Xu et al. (2020), Appendix C.
    pub fn is_splittable(&self, layer: &Layer) -> bool {
        layer.is_elementwise_activation()
            || matches!(layer, Layer::BilinearCrown(_) | Layer::MulBinary(_))
            // GenBaB norm branching (#norm-genbab): RmsNorm is splittable on its
            // internal inv_rms scalar even though inv_rms is not a graph node.
            || matches!(layer, Layer::RmsNorm(_))
    }

    /// Score a GenBaB norm-branching split of a `Layer::RmsNorm` node on its
    /// internal `inv_rms` scalar (#norm-genbab).
    ///
    /// The branchable range is the union (over normalization groups) of the
    /// per-group IBP-derived `inv_rms` interval, computed from the node's input
    /// bounds. We score by the range width (wider ⇒ looser reciprocal/sqrt
    /// relaxation ⇒ more to gain from splitting) and emit a single decision
    /// carrying the parent `inv_rms` range, which `to_splits` bisects.
    ///
    /// Returns `None` when the input bounds are unavailable, the range is
    /// degenerate / non-finite, or already narrower than `min_branch_width`.
    fn score_rms_norm_inv_rms(
        &self,
        node_name: &str,
        node: &crate::network::GraphNode,
        eps: f32,
        node_bounds: &HashMap<String, BoundedTensor>,
        applied_windows: Option<&[Option<(f32, f32)>]>,
    ) -> Result<Option<BranchingDecision>> {
        use crate::beta_crown::branching::LayerRef;

        let input_name = match node.inputs.first() {
            Some(n) => n.as_str(),
            None => return Ok(None),
        };
        // RmsNorm normalizes over the LAST axis; the input bounds live under the
        // input node name. (NETWORK_INPUT inputs are handled by the caller's
        // node_bounds map having that key, or are absent — then we skip.)
        let bounds = match node_bounds.get(input_name) {
            Some(b) => b,
            None => return Ok(None),
        };

        let shape = bounds.shape();
        let norm_size = match shape.last().copied() {
            Some(n) if n > 0 => n,
            _ => return Ok(None),
        };
        let flat_l = crate::contiguous_flat_slice(bounds.lower());
        let flat_u = crate::contiguous_flat_slice(bounds.upper());
        if flat_l.len() % norm_size != 0 {
            return Ok(None);
        }
        let groups = flat_l.len() / norm_size;

        // Pick the SINGLE group with the widest inv_rms interval to split.
        // Splitting one group at a time is required for soundness (a shared
        // window across groups creates a join gap between siblings; see
        // `InvRmsOverride`). The widest group has the loosest reciprocal/sqrt
        // relaxation, so it is the most valuable to subdivide.
        let mut best: Option<(usize, f32, f32, f32)> = None; // (group, lo, hi, width)
        for g in 0..groups {
            let xl = &flat_l[g * norm_size..(g + 1) * norm_size];
            let xu = &flat_u[g * norm_size..(g + 1) * norm_size];
            let (mut lo, mut hi) = inv_rms_interval(xl, xu, eps);
            // Intersect with the window already applied to THIS group on the BaB
            // path to this domain, so the candidate range shrinks each split and
            // branching converges (otherwise the input-derived range is constant).
            if let Some((wlo, whi)) = applied_windows.and_then(|w| w.get(g).copied().flatten()) {
                lo = lo.max(wlo);
                hi = hi.min(whi);
            }
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                continue;
            }
            let width = hi - lo;
            if best.map_or(true, |(_, _, _, bw)| width > bw) {
                best = Some((g, lo, hi, width));
            }
        }
        let (group, inv_lo, inv_hi, width) = match best {
            Some(b) => b,
            None => return Ok(None),
        };
        if width < self.config.min_branch_width {
            return Ok(None);
        }

        Ok(Some(BranchingDecision {
            layer: LayerRef::Name(node_name.to_string()),
            neuron_idx: 0,
            points: Vec::new(),
            // Score by inv_rms range width. The decorrelation penalty grows with
            // the relaxation gap, which grows with this width.
            score: width,
            original_bounds: (inv_lo, inv_hi),
            input_index: None,
            norm_inv_rms: Some((group, inv_lo, inv_hi)),
        }))
    }
}

/// IBP-derived `inv_rms = 1/sqrt(mean(x²)+eps)` interval for one normalization
/// group, matching the directed-rounded interval arithmetic in
/// `decomposed_rms_norm_crown_backward` (#norm-genbab).
fn inv_rms_interval(x_l: &[f32], x_u: &[f32], eps: f32) -> (f32, f32) {
    use crate::layers::normalization::math_common::square_interval_bounds;
    use ny_tensor::{next_down_f32, next_up_f32};
    let n = x_l.len();
    if n == 0 {
        return (f32::NAN, f32::NAN);
    }
    let nf = n as f64;
    let mut var_l = 0.0f64;
    let mut var_u = 0.0f64;
    for i in 0..n {
        let (sq_l, sq_u) = square_interval_bounds(x_l[i], x_u[i]);
        var_l += sq_l as f64;
        var_u += sq_u as f64;
    }
    let var_l = next_down_f32((var_l / nf) as f32);
    let var_u = next_up_f32((var_u / nf) as f32);
    let var_eps_l = next_down_f32((var_l as f64 + eps as f64) as f32);
    let var_eps_u = next_up_f32((var_u as f64 + eps as f64) as f32);
    let rms_l = next_down_f32((var_eps_l as f64).sqrt() as f32);
    let rms_u = next_up_f32((var_eps_u as f64).sqrt() as f32);
    (next_down_f32(1.0 / rms_u), next_up_f32(1.0 / rms_l))
}
