// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CachedLinearBounds helpers split out of `types.rs` to keep the type file under
//! the 500-line quality cap.

use std::collections::HashMap;
use std::time::Instant;

use ny_core::Result;
use ny_tensor::BoundedTensor;

use super::types::CachedLinearBounds;
use crate::LinearBounds;

impl CachedLinearBounds {
    /// #warm-cache-err: make one captured backward block publishable.
    ///
    /// `CachedLinearBounds` stores `(A, b)` only. A block that carries a
    /// certified coefficient error `E` (`LinearBounds::{lower,upper}_a_err`, the
    /// f32 rounding enclosure of a Linear backward's `A·W`) bounds the
    /// objective only with `E` charged: the true coefficient lies anywhere in
    /// `[A - E, A + E]`, so the bare `(A, b)` can sit up to
    /// `Σ_j E_ij·max|y_j|` on the wrong side of the objective, and a child
    /// warm-started from it can close on a bound that is not certified.
    ///
    /// This folds `E` OUTWARD into the bias over `node_box`, the certified
    /// enclosure of the values the block's columns multiply on the capturing
    /// domain (the node's own output box; the input box for a network-input
    /// block). It is the fold `spec_propagation/core.rs` already applied to
    /// its finite-deadline captures. Every region the cache is later read on
    /// (a descendant of the capturing domain) lies inside that box, so the
    /// folded block remains a bound there.
    ///
    /// Returns `Ok(true)` when the block is error-free (folded, or exact to
    /// begin with) and may be published. Returns `Ok(false)` when it cannot be
    /// folded soundly (no box, a box that does not match the columns, a
    /// malformed block, or a non-finite penalty): the caller must then leave
    /// the entry out, which only costs the warm start at that node. A deadline
    /// refusal during a finite fold is returned as `Err`.
    pub(crate) fn fold_capture_coeff_err(
        block: &mut LinearBounds,
        node_box: Option<&BoundedTensor>,
        deadline: Option<Instant>,
    ) -> Result<bool> {
        if !block.has_coeff_err() {
            return Ok(true);
        }
        let Some(node_box) = node_box else {
            return Ok(false);
        };
        if node_box.len() != block.num_inputs()
            || block.validate_internal_shapes().is_err()
            || block.validate_no_nan().is_err()
        {
            return Ok(false);
        }
        block.fold_coeff_err_over_box_eager_with_deadline(node_box, deadline)?;
        Ok(!block.has_coeff_err())
    }

    /// Stack single-row per-objective caches into one multi-row cache.
    ///
    /// Each input cache must contain one row per node in every stored A/bias
    /// entry. The resulting cache stores `caches.len()` rows per node, in the
    /// same order as the input slice.
    ///
    /// Returns `None` when the input is empty, any cache is empty, row counts
    /// differ from 1, or the per-node cache structure is inconsistent.
    pub fn stack_single_row(caches: &[&CachedLinearBounds]) -> Option<CachedLinearBounds> {
        if caches.is_empty() || caches.iter().any(|cache| cache.is_empty()) {
            return None;
        }

        let mut stacked = CachedLinearBounds {
            lower_a: stack_matrix_cache_rows(caches, |cache| &cache.lower_a)?,
            upper_a: stack_matrix_cache_rows(caches, |cache| &cache.upper_a)?,
            lower_b: stack_vector_cache_rows(caches, |cache| &cache.lower_b)?,
            upper_b: stack_vector_cache_rows(caches, |cache| &cache.upper_b)?,
            frontiers: HashMap::new(),
        };
        // #la-warm-frontier: only frontiers present, paired and of one
        // structure in EVERY row cache survive the stack, re-paired with the
        // stacked block; any other is dropped (its node then seeds under the
        // #warm-dag guard, never from a partial or foreign state).
        if caches.iter().any(|cache| cache.frontier_count() > 0) {
            stacked.frontiers = CachedLinearBounds::stacked_frontiers(caches, &stacked);
        }
        Some(stacked)
    }
}

fn stack_matrix_cache_rows(
    caches: &[&CachedLinearBounds],
    access: impl Fn(&CachedLinearBounds) -> &HashMap<String, ndarray::Array2<f32>>,
) -> Option<HashMap<String, ndarray::Array2<f32>>> {
    let first_map = access(*caches.first()?);
    if first_map.is_empty() {
        return None;
    }

    for cache in caches.iter().skip(1) {
        let map = access(cache);
        if map.len() != first_map.len() || !map.keys().all(|key| first_map.contains_key(key)) {
            return None;
        }
    }

    let mut stacked = HashMap::with_capacity(first_map.len());
    for (name, first_array) in first_map {
        if first_array.nrows() != 1 {
            return None;
        }
        let width = first_array.ncols();
        let mut data = Vec::with_capacity(caches.len() * width);
        for cache in caches {
            let array = access(cache).get(name)?;
            if array.nrows() != 1 || array.ncols() != width {
                return None;
            }
            data.extend(array.iter().copied());
        }
        let stacked_rows = ndarray::Array2::from_shape_vec((caches.len(), width), data).ok()?;
        stacked.insert(name.clone(), stacked_rows);
    }

    Some(stacked)
}

fn stack_vector_cache_rows(
    caches: &[&CachedLinearBounds],
    access: impl Fn(&CachedLinearBounds) -> &HashMap<String, ndarray::Array1<f32>>,
) -> Option<HashMap<String, ndarray::Array1<f32>>> {
    let first_map = access(*caches.first()?);
    if first_map.is_empty() {
        return None;
    }

    for cache in caches.iter().skip(1) {
        let map = access(cache);
        if map.len() != first_map.len() || !map.keys().all(|key| first_map.contains_key(key)) {
            return None;
        }
    }

    let mut stacked = HashMap::with_capacity(first_map.len());
    for (name, first_vector) in first_map {
        if first_vector.len() != 1 {
            return None;
        }
        let mut data = Vec::with_capacity(caches.len());
        for cache in caches {
            let vector = access(cache).get(name)?;
            if vector.len() != 1 {
                return None;
            }
            data.push(vector[0]);
        }
        stacked.insert(name.clone(), ndarray::Array1::from_vec(data));
    }

    Some(stacked)
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    use super::CachedLinearBounds;

    #[test]
    fn test_stack_single_row_round_trips_through_split_multi_row() {
        let mut row0 = CachedLinearBounds::default();
        row0.lower_a
            .insert("relu1".to_string(), arr2(&[[1.0, 2.0]]));
        row0.upper_a
            .insert("relu1".to_string(), arr2(&[[0.1, 0.2]]));
        row0.lower_b.insert("relu1".to_string(), arr1(&[10.0]));
        row0.upper_b.insert("relu1".to_string(), arr1(&[11.0]));

        let mut row1 = CachedLinearBounds::default();
        row1.lower_a
            .insert("relu1".to_string(), arr2(&[[3.0, 4.0]]));
        row1.upper_a
            .insert("relu1".to_string(), arr2(&[[0.3, 0.4]]));
        row1.lower_b.insert("relu1".to_string(), arr1(&[20.0]));
        row1.upper_b.insert("relu1".to_string(), arr1(&[21.0]));

        let stacked =
            CachedLinearBounds::stack_single_row(&[&row0, &row1]).expect("stack should succeed");
        assert_eq!(stacked.lower_a["relu1"], arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        assert_eq!(stacked.upper_b["relu1"], arr1(&[11.0, 21.0]));

        let split = stacked.split_multi_row(2).expect("split should succeed");
        assert_eq!(split.len(), 2);
        assert_eq!(split[0].lower_a["relu1"], row0.lower_a["relu1"]);
        assert_eq!(split[0].upper_b["relu1"], row0.upper_b["relu1"]);
        assert_eq!(split[1].lower_a["relu1"], row1.lower_a["relu1"]);
        assert_eq!(split[1].upper_b["relu1"], row1.upper_b["relu1"]);
    }

    #[test]
    fn test_stack_single_row_returns_none_for_mismatched_nodes() {
        let mut row0 = CachedLinearBounds::default();
        row0.lower_a
            .insert("relu1".to_string(), arr2(&[[1.0, 2.0]]));
        row0.upper_a
            .insert("relu1".to_string(), arr2(&[[0.1, 0.2]]));
        row0.lower_b.insert("relu1".to_string(), arr1(&[10.0]));
        row0.upper_b.insert("relu1".to_string(), arr1(&[11.0]));

        let mut row1 = CachedLinearBounds::default();
        row1.lower_a
            .insert("relu2".to_string(), arr2(&[[3.0, 4.0]]));
        row1.upper_a
            .insert("relu2".to_string(), arr2(&[[0.3, 0.4]]));
        row1.lower_b.insert("relu2".to_string(), arr1(&[20.0]));
        row1.upper_b.insert("relu2".to_string(), arr1(&[21.0]));

        assert!(CachedLinearBounds::stack_single_row(&[&row0, &row1]).is_none());
    }

    fn exact_block() -> crate::LinearBounds {
        crate::LinearBounds::new(
            arr2(&[[-1.0, 1.0]]),
            arr1(&[0.0]),
            arr2(&[[-1.0, 1.0]]),
            arr1(&[0.0]),
        )
        .expect("finite block")
    }

    /// The same coefficients with a certified coefficient error: the true
    /// coefficients lie anywhere in `[A - 1e-3, A + 1e-3]`, so over `y = (1, 1)`
    /// the certified lower plane is `-2e-3`, not the bare `0`.
    fn error_carrying_block() -> crate::LinearBounds {
        let mut block = exact_block();
        block.set_coeff_err(arr2(&[[1e-3, 1e-3]]), arr2(&[[1e-3, 1e-3]]));
        block
    }

    /// #warm-cache-err backstop: `from_linear_bounds_map` has no box to fold a
    /// certified coefficient error over, so a block that still carries one
    /// must not be published as the bare `(A, b)`, which is tighter than the
    /// certified bound.
    #[test]
    fn warm_err_from_linear_bounds_map_never_publishes_a_bare_error_carrying_block() {
        let map = std::collections::HashMap::from([
            ("carrying".to_string(), error_carrying_block()),
            ("exact".to_string(), exact_block()),
        ]);
        let cache = CachedLinearBounds::from_linear_bounds_map(map);
        assert!(
            cache.linear_bounds("exact").is_some(),
            "an error-free block must still be published"
        );
        assert!(
            cache.linear_bounds("carrying").is_none(),
            "an error-carrying block was published without its error charge"
        );
    }

    /// #warm-cache-err backstop: `override_node` likewise has no box, so an
    /// error-carrying replacement must leave the existing (sound) entry alone.
    #[test]
    fn warm_err_override_node_never_installs_a_bare_error_carrying_block() {
        let fallback = crate::LinearBounds::conservative(1, 2);
        let mut cache = CachedLinearBounds::from_linear_bounds_map(
            std::collections::HashMap::from([("node".to_string(), fallback)]),
        );
        let before = (
            cache.lower_a["node"].clone(),
            cache.lower_b["node"].clone(),
            cache.upper_a["node"].clone(),
            cache.upper_b["node"].clone(),
        );
        let _ = cache.override_node("node", error_carrying_block());
        let after = (
            cache.lower_a["node"].clone(),
            cache.lower_b["node"].clone(),
            cache.upper_a["node"].clone(),
            cache.upper_b["node"].clone(),
        );
        assert_eq!(
            before, after,
            "override_node installed an error-carrying block without its error charge"
        );
    }
}
