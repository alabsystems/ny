// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for multi-objective graph BaB orchestration.
//!
//! Extracted from `multi_objective.rs` to keep the top-level verification loop
//! focused on queue/domain flow while preserving behavior.

use ndarray::Array2;
use ny_tensor::BoundedTensor;

use crate::batched_domain::CachedLinearBounds;

/// Batteries-included gate for the multi-objective GPU single-pass domain lane
/// (#w5-bab-throughput): route beta-opt-eligible BaB children through the
/// domain-batched single-pass adapter (whole-suffix sound GPU backward with the
/// inherited-β dual folded) instead of the ~3s-per-pass CPU per-child beta-opt,
/// and prune the adapter's dense spec matrix to the union of unverified
/// objectives. ON by default; opt out with `NY_MO_GPU_BATCH=0` (disable-flag
/// principle) to restore the legacy per-child lane byte-identically.
pub(in crate::beta_crown::engine::graph) fn multi_objective_gpu_single_pass_enabled() -> bool {
    !matches!(std::env::var("NY_MO_GPU_BATCH").ok().as_deref(), Some("0"))
}

/// Per-domain β OPTIMIZATION inside the GPU single-pass lane
/// (#w4-split-tightening). Default ON; `NY_MO_GPU_BETA=0` restores the
/// single-shot (inherited-β) GPU lane byte-identically.
pub(in crate::beta_crown::engine::graph) fn multi_objective_gpu_beta_enabled() -> bool {
    !matches!(std::env::var("NY_MO_GPU_BETA").ok().as_deref(), Some("0"))
}

/// Unverified objective subset for child-domain propagation.
///
/// alpha-beta-CROWN prunes verified OR-specs before later optimization passes
/// (`complete_verifier/prune.py:27-98`,
/// `complete_verifier/incomplete_verifier_func.py:277-370`). Ny keeps the
/// full bound vector on each domain for queue accounting, but child CROWN only
/// needs to revisit the still-unverified objectives.
pub(crate) struct PrunedMultiObjectiveTargets {
    pub(crate) active_indices: Vec<usize>,
    pub(crate) objectives: Vec<Vec<f32>>,
    pub(crate) thresholds: Vec<f32>,
    pub(crate) verified_mask: Vec<bool>,
}

/// Keep only the objectives that are not yet verified in the current domain.
pub(crate) fn prune_verified_multi_objective_targets(
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    verified_mask: &[bool],
) -> PrunedMultiObjectiveTargets {
    debug_assert_eq!(
        objectives.len(),
        thresholds.len(),
        "prune_verified_multi_objective_targets(): objectives/thresholds mismatch"
    );
    debug_assert_eq!(
        objectives.len(),
        verified_mask.len(),
        "prune_verified_multi_objective_targets(): objectives/verified_mask mismatch"
    );

    let mut active_indices = Vec::new();
    let mut active_objectives = Vec::new();
    let mut active_thresholds = Vec::new();

    for (idx, ((objective, &threshold), &verified)) in objectives
        .iter()
        .zip(thresholds.iter())
        .zip(verified_mask.iter())
        .enumerate()
    {
        if verified {
            continue;
        }
        active_indices.push(idx);
        active_objectives.push(objective.clone());
        active_thresholds.push(threshold);
    }

    let active_verified_mask = vec![false; active_indices.len()];
    PrunedMultiObjectiveTargets {
        active_indices,
        objectives: active_objectives,
        thresholds: active_thresholds,
        verified_mask: active_verified_mask,
    }
}

/// Merge updated bounds for the active objective subset back into the full vector.
pub(crate) fn merge_pruned_objective_bounds(
    full_bounds: &[(f32, f32)],
    pruned_targets: &PrunedMultiObjectiveTargets,
    active_bounds: Vec<(f32, f32)>,
) -> Vec<(f32, f32)> {
    debug_assert_eq!(
        pruned_targets.active_indices.len(),
        active_bounds.len(),
        "merge_pruned_objective_bounds(): active index/bounds mismatch"
    );

    let mut merged_bounds = full_bounds.to_vec();
    for (idx, bounds) in pruned_targets
        .active_indices
        .iter()
        .copied()
        .zip(active_bounds)
    {
        merged_bounds[idx] = bounds;
    }
    merged_bounds
}

/// Select per-objective cached lA entries for the active objective subset.
pub(crate) fn prune_cached_las_for_targets<'a>(
    full_cached_las: &'a [Option<CachedLinearBounds>],
    pruned_targets: &PrunedMultiObjectiveTargets,
) -> Vec<Option<&'a CachedLinearBounds>> {
    pruned_targets
        .active_indices
        .iter()
        .map(|&idx| full_cached_las.get(idx).and_then(Option::as_ref))
        .collect()
}

/// Merge updated caches for the active objective subset back into the full vector.
pub(crate) fn merge_pruned_cached_las(
    full_cached_las: &[Option<CachedLinearBounds>],
    pruned_targets: &PrunedMultiObjectiveTargets,
    active_cached_las: Vec<Option<CachedLinearBounds>>,
) -> Vec<Option<CachedLinearBounds>> {
    debug_assert_eq!(
        pruned_targets.active_indices.len(),
        active_cached_las.len(),
        "merge_pruned_cached_las(): active index/cache mismatch"
    );

    let mut merged_cached_las = full_cached_las.to_vec();
    for (idx, cache) in pruned_targets
        .active_indices
        .iter()
        .copied()
        .zip(active_cached_las)
    {
        merged_cached_las[idx] = cache;
    }
    merged_cached_las
}

/// Build a dense specification matrix for spec-guided CROWN.
///
/// Returns `None` when objective dimensions are inconsistent.
///
/// Visibility widened to `pub(in crate::beta_crown::engine::graph)` so the
/// domain-batched single-pass adapter (`batched::batched_dense_specs`) can build
/// one uniform spec matrix from the full objective set (#perf).
pub(in crate::beta_crown::engine::graph) fn build_spec_matrix(
    objectives: &[Vec<f32>],
) -> Option<Array2<f32>> {
    if objectives.is_empty() {
        return None;
    }
    let num_specs = objectives.len();
    let output_dim = objectives[0].len();
    let mut data = Vec::with_capacity(num_specs * output_dim);
    for obj in objectives {
        if obj.len() != output_dim {
            return None;
        }
        data.extend_from_slice(obj);
    }
    Array2::from_shape_vec((num_specs, output_dim), data).ok()
}

/// Convert scalar spec bounds to `(lower, upper)` tuples.
///
/// Visibility widened to `pub(in crate::beta_crown::engine::graph)` for the
/// domain-batched single-pass adapter (`batched::batched_dense_specs`).
pub(in crate::beta_crown::engine::graph) fn spec_bounds_to_vec(
    bounds: &BoundedTensor,
) -> Vec<(f32, f32)> {
    let flat = bounds.flatten();
    (0..flat.len())
        .map(|i| (flat.lower()[[i]], flat.upper()[[i]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        merge_pruned_cached_las, merge_pruned_objective_bounds, prune_cached_las_for_targets,
        prune_verified_multi_objective_targets,
    };
    use crate::batched_domain::CachedLinearBounds;

    #[test]
    fn test_prune_verified_multi_objective_targets_keeps_unverified_order_3813() {
        let objectives = vec![vec![1.0], vec![2.0], vec![3.0]];
        let thresholds = vec![0.1, 0.2, 0.3];
        let verified_mask = vec![true, false, true];

        let pruned =
            prune_verified_multi_objective_targets(&objectives, &thresholds, &verified_mask);

        assert_eq!(pruned.active_indices, vec![1]);
        assert_eq!(pruned.objectives, vec![vec![2.0]]);
        assert_eq!(pruned.thresholds, vec![0.2]);
        assert_eq!(pruned.verified_mask, vec![false]);
    }

    #[test]
    fn test_merge_pruned_objective_bounds_restores_verified_slots_3813() {
        let objectives = vec![vec![1.0], vec![2.0], vec![3.0]];
        let thresholds = vec![0.1, 0.2, 0.3];
        let verified_mask = vec![true, false, true];
        let full_bounds = vec![(10.0, 11.0), (20.0, 21.0), (30.0, 31.0)];

        let pruned =
            prune_verified_multi_objective_targets(&objectives, &thresholds, &verified_mask);
        let merged = merge_pruned_objective_bounds(&full_bounds, &pruned, vec![(4.0, 5.0)]);

        assert_eq!(merged, vec![(10.0, 11.0), (4.0, 5.0), (30.0, 31.0)]);
    }

    #[test]
    fn test_prune_cached_las_for_targets_keeps_active_alignment_3813() {
        let objectives = vec![vec![1.0], vec![2.0], vec![3.0]];
        let thresholds = vec![0.1, 0.2, 0.3];
        let verified_mask = vec![false, true, false];
        let pruned =
            prune_verified_multi_objective_targets(&objectives, &thresholds, &verified_mask);

        let mut cache0 = CachedLinearBounds::default();
        cache0
            .lower_b
            .insert("relu0".to_string(), ndarray::arr1(&[1.0]));
        let mut cache2 = CachedLinearBounds::default();
        cache2
            .lower_b
            .insert("relu2".to_string(), ndarray::arr1(&[3.0]));

        let full_cached_las = vec![Some(cache0), None, Some(cache2)];
        let active_cached_las = prune_cached_las_for_targets(&full_cached_las, &pruned);

        assert_eq!(active_cached_las.len(), 2);
        assert_eq!(
            active_cached_las[0]
                .and_then(|cache| cache.lower_b.get("relu0"))
                .map(|bias| bias[0]),
            Some(1.0)
        );
        assert_eq!(
            active_cached_las[1]
                .and_then(|cache| cache.lower_b.get("relu2"))
                .map(|bias| bias[0]),
            Some(3.0)
        );
    }

    #[test]
    fn test_merge_pruned_cached_las_restores_verified_slots_3813() {
        let objectives = vec![vec![1.0], vec![2.0], vec![3.0]];
        let thresholds = vec![0.1, 0.2, 0.3];
        let verified_mask = vec![true, false, true];
        let pruned =
            prune_verified_multi_objective_targets(&objectives, &thresholds, &verified_mask);

        let mut full_cache0 = CachedLinearBounds::default();
        full_cache0
            .lower_b
            .insert("relu0".to_string(), ndarray::arr1(&[10.0]));
        let mut active_cache = CachedLinearBounds::default();
        active_cache
            .lower_b
            .insert("relu1".to_string(), ndarray::arr1(&[20.0]));

        let merged = merge_pruned_cached_las(
            &[Some(full_cache0.clone()), None, None],
            &pruned,
            vec![Some(active_cache.clone())],
        );

        assert_eq!(merged.len(), 3);
        assert_eq!(
            merged[0]
                .as_ref()
                .and_then(|cache| cache.lower_b.get("relu0"))
                .map(|bias| bias[0]),
            Some(10.0)
        );
        assert_eq!(
            merged[1]
                .as_ref()
                .and_then(|cache| cache.lower_b.get("relu1"))
                .map(|bias| bias[0]),
            Some(20.0)
        );
        assert!(merged[2].is_none());
    }
}
