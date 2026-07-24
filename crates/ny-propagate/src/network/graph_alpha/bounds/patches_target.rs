// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) fn patches_dense_fallback_details(
    bounds: &CrownBounds,
    site: &'static str,
) -> Result<Option<String>> {
    let CrownBounds::Patches(pb) = bounds else {
        return Ok(None);
    };
    let (rows, cols) = pb.dense_pair_shape()?;
    let budget = cpu_crown_dense_budget_bytes();
    let estimate = DenseMaterializationEstimate {
        site,
        rows,
        cols,
        required_bytes: pb.dense_pair_bytes()?,
    };
    if estimate.exceeds_budget(budget) {
        Ok(Some(estimate.budget_exceeded_details(budget)))
    } else {
        Ok(None)
    }
}

impl GraphNetwork {
    /// Flat input size (`in_channels * input_h * input_w`) of the deepest
    /// Conv2d ancestor of `node_name` (the one closest to the network input,
    /// i.e. first in topological order). Returns `None` when no Conv2d ancestor
    /// exists or the conv has no recorded input shape.
    ///
    /// This is the column count of the `[target_dim x conv_in_size]` BACKWARD
    /// coefficient pair — the pair that actually OOMs the dense path, as opposed
    /// to the `[target_dim x target_dim]` identity pair (#patches-backward-oom).
    fn deepest_conv_ancestor_in_size(&self, node_name: &str) -> Option<usize> {
        let relevant_nodes = self.ancestors(node_name).ok()?;
        // Topological order: dependencies before dependents, so the first Conv2d
        // encountered is the deepest ancestor (closest to the input).
        for name in &relevant_nodes {
            if let Some(node) = self.nodes.get(name) {
                match &node.layer {
                    Layer::Conv2d(conv) => {
                        let (in_h, in_w) = conv.input_shape?;
                        return conv
                            .in_channels()
                            .checked_mul(in_h)
                            .and_then(|chw| chw.checked_mul(in_w));
                    }
                    // Stage 2a: a STRIDE-1 ConvTranspose2d ancestor is patches-
                    // eligible too (its backward pair is `[target_dim x conv_in]`,
                    // conv_in = in_channels·in_h·in_w). Derive the same column
                    // count so mid-size ConvTranspose targets route to the exact
                    // patches bound instead of the loose-IBP fallback. Stride>1
                    // (and other unsupported corners) fall back to dense at the
                    // ConvTranspose layer, so admitting them here would give no
                    // memory benefit and only perturb routing — skip them (the
                    // pre-stage-2a behavior).
                    Layer::ConvTranspose2d(convt) if convt.stride == (1, 1) => {
                        let (in_h, in_w) = convt.input_shape?;
                        return convt
                            .in_channels()
                            .checked_mul(in_h)
                            .and_then(|chw| chw.checked_mul(in_w));
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// Test-only accessor for the patches-target predicate so soundness tests
    /// can assert `is_patches_target` directly (#patches-backward-oom).
    #[cfg(test)]
    pub(crate) fn crown_ibp_target_can_start_in_patches_for_test(
        &self,
        node_name: &str,
        bounds: &BoundedTensor,
    ) -> bool {
        self.crown_ibp_target_can_start_in_patches(node_name, bounds)
    }

    pub(super) fn crown_ibp_target_can_start_in_patches(
        &self,
        node_name: &str,
        bounds: &BoundedTensor,
    ) -> bool {
        let budget = cpu_crown_dense_budget_bytes();
        let target_dim = bounds.len();
        let dense_identity_exceeds_budget =
            match crate::network::crown_memory::identity_pair_bytes(target_dim) {
                Some(required) => required > budget,
                None => true,
            };

        // The dense path OOMs on the [target_dim x conv_in_size] BACKWARD pair,
        // not just the [target_dim x target_dim] IDENTITY pair. Admit patches
        // whenever EITHER pair exceeds budget: routing mid-size conv targets
        // (identity fits, backward pair does not) to the EXACT patches bound is
        // strictly tighter-or-equal than the loose-IBP fallback they take today
        // (bit-equivalent-to-dense within 1e-5, proven by crown_patches.rs:29).
        // Reuses dense_pair_bytes / the ancestor walk already in this file.
        let dense_backward_exceeds_budget = self
            .deepest_conv_ancestor_in_size(node_name)
            .and_then(|conv_in_size| {
                crate::network::crown_memory::dense_pair_bytes(target_dim, conv_in_size)
            })
            .map(|required| required > budget)
            .unwrap_or(false);

        (dense_identity_exceeds_budget || dense_backward_exceeds_budget)
            && bounds.shape().len() == 3
            && self
                .nodes
                .get(node_name)
                .is_some_and(|node| node.inputs.len() == 1)
            && self
                .ancestors(node_name)
                .map(|relevant_nodes| {
                    relevant_nodes.iter().any(|name| {
                        self.nodes.get(name).is_some_and(|node| {
                            // Only patches-eligible conv ancestors: any Conv2d, or
                            // a STRIDE-1 ConvTranspose2d (stage 2a). Stride>1
                            // ConvTranspose falls back to dense, so it must not
                            // flip a target into patches mode (pre-stage-2a route).
                            match &node.layer {
                                Layer::Conv2d(_) => true,
                                Layer::ConvTranspose2d(ct) => ct.stride == (1, 1),
                                _ => false,
                            }
                        })
                    })
                })
                .unwrap_or(false)
    }
}
