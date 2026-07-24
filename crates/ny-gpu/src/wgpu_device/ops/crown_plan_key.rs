// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN plan cache key computation, split from crown_plan.rs for size.
//!
//! # Static-data key: pointer identity, not content (#perf-plan-cache)
//!
//! Model weights (`Linear::weight`, `Linear::bias`, `Conv2d::weight_col`,
//! `Conv2d::bias_expanded`) are stored as `Arc<[f32]>` and are *stable* across
//! BaB domains: the same `Arc` allocation is shared by every CROWN backward
//! call for a given model. Hashing every `f32` of every weight on each call
//! cost ~3.7 ms / 10 MB (~65 ms / 100 MB for vggnet16) and was pure redundant
//! work — the bytes never change for a live `Arc`.
//!
//! Instead we key the static-data hash on each weight `Arc`'s **pointer
//! identity** (`Arc::as_ptr()` data address) plus its length. This makes the
//! lookup O(num_layers) rather than O(weight_bytes), with identical cache
//! semantics:
//!
//! * Same `Arc` → same (pointer, len) → same key → same cached plan (true hit).
//! * Two *distinct* `Arc<[f32]>` allocations have distinct data pointers, so
//!   they map to distinct keys — no false cache hit, even if their contents
//!   happen to be bit-identical. (Two weight tensors that share one `Arc` are,
//!   by construction, the same tensor and correctly collide.)
//!
//! ## Soundness
//!
//! This changes only *how the cache key is computed*, never *which plan is
//! selected for a given set of weights*, and never any computed bound. A
//! cached `PreparedCrownPlan` only captures static topology + buffer layout
//! (dynamic per-domain slopes/intercepts are re-uploaded every call via
//! `refresh_crown_plan_dynamic_layers`), so reusing a plan keyed by pointer
//! identity is exactly as correct as reusing one keyed by content hash.
//!
//! ## Lifetime invariant (ENFORCED, not assumed)
//!
//! Pointer identity is only meaningful while the `Arc` is alive. This is now
//! enforced structurally: every cached `PreparedCrownPlan` holds keep-alive
//! clones of the weight `Arc`s it was keyed by (`static_weight_arcs`,
//! crown_plan.rs), so an address can never be recycled by the allocator while
//! a key hashing that address is live — a pointer-identity hit is therefore
//! always a true content hit. (Before this, dropping one model's weights and
//! loading another could recycle the same allocation address and silently
//! serve the *old* model's cached plan — observed as
//! `test_crown_backward_dual_alpha_crossing` returning the previous test's
//! bounds.) The VNN-COMP runner still calls
//! [`WgpuDevice::clear_crown_plan_cache`] between models (see crown_plan.rs)
//! to release the retained weight memory alongside the GPU buffers.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ny_core::GpuCrownLayer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CrownPlanKey {
    topology_hash: u64,
    static_data_hash: u64,
    num_specs: usize,
    first_dim: usize,
}

pub(crate) fn crown_plan_key(
    layers: &[GpuCrownLayer],
    num_specs: usize,
    first_dim: usize,
) -> CrownPlanKey {
    let mut topology = DefaultHasher::new();
    let mut static_data = DefaultHasher::new();

    for layer in layers {
        match layer {
            GpuCrownLayer::Linear {
                weight,
                bias,
                out_features,
                in_features,
            } => {
                0u8.hash(&mut topology);
                out_features.hash(&mut topology);
                in_features.hash(&mut topology);
                bias.is_some().hash(&mut topology);

                out_features.hash(&mut static_data);
                in_features.hash(&mut static_data);
                hash_arc_identity(&mut static_data, weight);
                bias.is_some().hash(&mut static_data);
                if let Some(bias) = bias {
                    hash_arc_identity(&mut static_data, bias);
                }
            }
            GpuCrownLayer::Activation { num_neurons, .. } => {
                1u8.hash(&mut topology);
                num_neurons.hash(&mut topology);
            }
            GpuCrownLayer::ActivationReluDualAlpha { num_neurons, .. } => {
                4u8.hash(&mut topology);
                num_neurons.hash(&mut topology);
            }
            GpuCrownLayer::MaxPool2d {
                input_dim,
                output_dim,
                ..
            } => {
                2u8.hash(&mut topology);
                input_dim.hash(&mut topology);
                output_dim.hash(&mut topology);
            }
            GpuCrownLayer::Conv2d {
                weight_col,
                bias_expanded,
                out_channels,
                in_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                out_h,
                out_w,
                in_h,
                in_w,
            } => {
                3u8.hash(&mut topology);
                out_channels.hash(&mut topology);
                in_channels.hash(&mut topology);
                kernel_h.hash(&mut topology);
                kernel_w.hash(&mut topology);
                stride_h.hash(&mut topology);
                stride_w.hash(&mut topology);
                pad_h.hash(&mut topology);
                pad_w.hash(&mut topology);
                out_h.hash(&mut topology);
                out_w.hash(&mut topology);
                in_h.hash(&mut topology);
                in_w.hash(&mut topology);
                bias_expanded.is_some().hash(&mut topology);

                out_channels.hash(&mut static_data);
                in_channels.hash(&mut static_data);
                kernel_h.hash(&mut static_data);
                kernel_w.hash(&mut static_data);
                stride_h.hash(&mut static_data);
                stride_w.hash(&mut static_data);
                pad_h.hash(&mut static_data);
                pad_w.hash(&mut static_data);
                out_h.hash(&mut static_data);
                out_w.hash(&mut static_data);
                in_h.hash(&mut static_data);
                in_w.hash(&mut static_data);
                hash_arc_identity(&mut static_data, weight_col);
                bias_expanded.is_some().hash(&mut static_data);
                if let Some(bias_expanded) = bias_expanded {
                    hash_arc_identity(&mut static_data, bias_expanded);
                }
            }
        }
    }

    CrownPlanKey {
        topology_hash: topology.finish(),
        static_data_hash: static_data.finish(),
        num_specs,
        first_dim,
    }
}

/// Hash a static-weight `Arc<[f32]>` by **pointer identity + length** instead
/// of by content (#perf-plan-cache).
///
/// `Arc::as_ptr` yields the address of the shared `[f32]` allocation. Two
/// distinct `Arc` allocations have distinct data addresses, so this never
/// produces a false cache hit between different weight tensors; the same `Arc`
/// always hashes identically, so genuine reuse hits the cache. Length is mixed
/// in as well so the key reflects the tensor's shape and to make collisions
/// between recycled addresses (after the cache is cleared between models)
/// vanishingly unlikely. This is O(1) per weight rather than O(weight_bytes).
///
/// See the module-level docs for the cache-invalidation invariant this relies
/// on.
fn hash_arc_identity(hasher: &mut DefaultHasher, values: &Arc<[f32]>) {
    // Address of the shared allocation: identity is stable for the lifetime
    // of the Arc and distinct across distinct allocations. `Arc::as_ptr`
    // returns a `*const [f32]` fat pointer; `.cast::<f32>()` discards the
    // length metadata, leaving just the data address.
    let ptr = Arc::as_ptr(values).cast::<f32>() as usize;
    ptr.hash(hasher);
    values.len().hash(hasher);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear(weight: Arc<[f32]>, bias: Option<Arc<[f32]>>) -> Vec<GpuCrownLayer> {
        // 2x3 weight, optional 2-element bias; shapes fixed so only the
        // weight/bias Arc identity varies between cases.
        vec![GpuCrownLayer::Linear {
            weight,
            bias,
            out_features: 2,
            in_features: 3,
        }]
    }

    /// Same `Arc` (shared allocation) must produce the same key → true hit.
    #[test]
    fn same_arc_same_key() {
        let weight: Arc<[f32]> = Arc::from(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let bias: Arc<[f32]> = Arc::from(vec![0.5_f32, -0.5]);

        let a = crown_plan_key(&linear(weight.clone(), Some(bias.clone())), 4, 2);
        let b = crown_plan_key(&linear(weight, Some(bias)), 4, 2);

        assert_eq!(
            a, b,
            "the same weight/bias Arc must hash to the same plan key (cache hit)"
        );
    }

    /// `Arc::clone` shares the allocation, so cloning the layer's Arc must not
    /// recompute or perturb the key — distinct from a fresh allocation below.
    #[test]
    fn cloned_arc_same_key() {
        let weight: Arc<[f32]> = Arc::from(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let cloned = Arc::clone(&weight);

        let a = crown_plan_key(&linear(weight, None), 4, 2);
        let b = crown_plan_key(&linear(cloned, None), 4, 2);

        assert_eq!(a, b, "Arc::clone shares the allocation: keys must match");
    }

    /// A *different* `Arc` allocation with bit-identical contents must produce
    /// a *different* key — no false cache hit. This is the core soundness
    /// property of pointer-identity keying.
    #[test]
    fn different_arc_same_contents_different_key() {
        let contents = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight_a: Arc<[f32]> = Arc::from(contents.clone());
        let weight_b: Arc<[f32]> = Arc::from(contents);
        // Sanity: genuinely distinct allocations.
        assert_ne!(
            Arc::as_ptr(&weight_a).cast::<f32>() as usize,
            Arc::as_ptr(&weight_b).cast::<f32>() as usize,
            "test precondition: the two Arcs must be distinct allocations"
        );

        let a = crown_plan_key(&linear(weight_a, None), 4, 2);
        let b = crown_plan_key(&linear(weight_b, None), 4, 2);

        assert_ne!(
            a, b,
            "distinct weight Arcs must not collide even with identical contents"
        );
    }

    /// Genuinely different weights (different Arc, different contents) must
    /// produce different keys.
    #[test]
    fn different_weights_different_key() {
        let weight_a: Arc<[f32]> = Arc::from(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let weight_b: Arc<[f32]> = Arc::from(vec![6.0_f32, 5.0, 4.0, 3.0, 2.0, 1.0]);

        let a = crown_plan_key(&linear(weight_a, None), 4, 2);
        let b = crown_plan_key(&linear(weight_b, None), 4, 2);

        assert_ne!(a, b, "different weight tensors must map to different keys");
    }

    /// Changing the dynamic dimensions (num_specs / first_dim) must still
    /// change the key — the optimization must not collapse those distinctions.
    #[test]
    fn dims_participate_in_key() {
        let weight: Arc<[f32]> = Arc::from(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let base = crown_plan_key(&linear(weight.clone(), None), 4, 2);
        let other_specs = crown_plan_key(&linear(weight.clone(), None), 8, 2);
        let other_first = crown_plan_key(&linear(weight, None), 4, 3);

        assert_ne!(base, other_specs, "num_specs must affect the key");
        assert_ne!(base, other_first, "first_dim must affect the key");
    }

    /// Presence vs. absence of a bias must change the key (topology change).
    #[test]
    fn bias_presence_affects_key() {
        let weight: Arc<[f32]> = Arc::from(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let bias: Arc<[f32]> = Arc::from(vec![0.5_f32, -0.5]);

        let with_bias = crown_plan_key(&linear(weight.clone(), Some(bias)), 4, 2);
        let without_bias = crown_plan_key(&linear(weight, None), 4, 2);

        assert_ne!(
            with_bias, without_bias,
            "bias presence must be reflected in the key"
        );
    }
}
