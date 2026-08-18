// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Fraction of the CPU dense budget above which a target's dense backward pair
/// is treated as expensive enough to prefer the patches representation, even
/// though it still fits (#conv-crown-residual).
///
/// At the 2 GiB default this puts the crossover at 128 MiB per pair — large
/// enough that small conv targets (oval21, mnist_fc-scale) keep their existing
/// dense route byte-for-byte, small enough to catch the mid-size ResNet conv
/// targets that dominate the cifar100/tinyimagenet gap.
const PATCHES_COST_ADMISSION_DIVISOR: usize = 16;

/// Absolute byte threshold above which a target's dense backward pair is
/// treated as expensive enough to prefer patches (#conv-crown-residual).
///
/// #threshold-vs-adaptive-budget: this is deliberately denominated against the
/// FIXED `DEFAULT_CROWN_DENSE_BUDGET_MB`, not against the live
/// `cpu_crown_dense_budget_bytes()`. The divisor was calibrated on 2026-07-27
/// against a budget that was then a hard-coded 2 GiB, and its docstring pins the
/// intended crossover at 128 MiB per pair. On 2026-07-29 the dense budget became
/// HOST-ADAPTIVE (`clamp(observed/2, 2 GiB, 12 GiB)`), which silently dragged
/// this crossover along with it: on a 24 GiB host the budget resolves to 6 GiB
/// and the crossover moved 128 MiB -> 402 MiB, i.e. the admission got 3x LOOSER
/// on exactly the machines with more memory to spend.
///
/// Measured consequence: CIFAR100_resnet_medium's largest demanded target
/// (target_dim 14400 over conv_in_size 3072) has a dense pair of 353_894_400 B.
/// Under the calibrated 128 MiB crossover it is admitted to patches; under the
/// drifted 402 MiB one it is not — which is precisely the state this constant
/// was added to fix (its own comment: "every target takes the slow dense path
/// and the residual-Add patches route above is DEAD CODE on that whole
/// benchmark"). A cost threshold must not move when an unrelated MEMORY budget
/// learns to measure its host.
pub(super) fn patches_cost_admission_threshold_bytes() -> usize {
    (crate::network::crown_memory::DEFAULT_CROWN_DENSE_BUDGET_MB * 1024 * 1024)
        / patches_cost_admission_divisor()
}

/// Experiment override for the cost-admission threshold
/// (`NY_PATCHES_COST_DIVISOR`). Absent ⇒ the compiled default, byte-identical.
fn patches_cost_admission_divisor() -> usize {
    std::env::var("NY_PATCHES_COST_DIVISOR")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|d| *d > 0)
        .unwrap_or(PATCHES_COST_ADMISSION_DIVISOR)
}

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

        // COST admission (#conv-crown-residual). Both conditions above are
        // *OOM* triggers: they admit patches only once the dense pair no longer
        // FITS. But dense is the wrong choice long before it stops fitting —
        // patches are 50-500x cheaper wherever the composed receptive field is
        // still small, and the CPU dense conv backward is the measured
        // bottleneck (the relusplitter preset records 257.56 s CPU vs 12.97 s
        // GPU for exactly this GEMM).
        //
        // The consequence of admitting on OOM alone is stark: on
        // `CIFAR100_resnet_medium` NOT ONE of the 11 demanded targets qualifies
        // at the default 2 GiB budget (largest is target_dim 14400 → a 1.659 GB
        // identity pair and a 0.354 GB backward pair), so every target takes the
        // slow dense path and the residual-`Add` patches route above is dead
        // code on that whole benchmark.
        //
        // So also admit when the dense backward pair is merely EXPENSIVE. This
        // is safe in both directions:
        // - Tightness: for many-row seeds the patches bound is bit-equivalent to
        //   dense within 1e-5 (`crown_patches.rs:29`). The one case where dense
        //   is genuinely tighter — thin seeds through overlapping receptive
        //   fields (#cgan-alpha-on-tight-refs) — is excluded by the same
        //   `patches_reentry_min_rows()` floor the Dense→Patches re-entry uses.
        // - Cost: if patches stop paying off mid-walk, the per-step crossover
        //   (`would_conv_compose_cover_input`, `patches_step.rs:327`) converts to
        //   dense at exactly that point, so an early patches start is
        //   self-correcting rather than a commitment.
        //
        // This is a pure widening: every target admitted before is still
        // admitted.
        let dense_backward_cost_prefers_patches = target_dim
            >= crate::network::core::graph::backward_helpers::patches_reentry_min_rows()
            && self
                .deepest_conv_ancestor_in_size(node_name)
                .and_then(|conv_in_size| {
                    crate::network::crown_memory::dense_pair_bytes(target_dim, conv_in_size)
                })
                .map(|required| required > patches_cost_admission_threshold_bytes())
                .unwrap_or(false);

        // The walk starts AT the target (`ancestors()` is inclusive, see
        // `traversal.rs:60`), so its first backward step crosses the target
        // node's own layer — the target must therefore be a node the patches
        // step can consume. Historically that meant strictly single-input,
        // which excluded every residual `Add` from ever seeding in patches. On a
        // ResNet the demanded pre-activation targets frequently ARE the residual
        // `Add`s, so that exclusion densified them unconditionally
        // (#conv-crown-residual, docs/PATCHES_RESIDUAL_ADD_ROOT_CAUSE_2026-07-27.md).
        (dense_identity_exceeds_budget
            || dense_backward_exceeds_budget
            || dense_backward_cost_prefers_patches)
            && bounds.shape().len() == 3
            && self.nodes.get(node_name).is_some_and(
                crate::network::core::graph::backward_helpers::node_admits_patches_backward_step,
            )
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
