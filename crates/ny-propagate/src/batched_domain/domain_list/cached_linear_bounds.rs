// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CachedLinearBounds helpers split out of `types.rs` to keep the type file under
//! the 500-line quality cap.

use std::collections::HashMap;

use super::types::CachedLinearBounds;

impl CachedLinearBounds {
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

        Some(CachedLinearBounds {
            lower_a: stack_matrix_cache_rows(caches, |cache| &cache.lower_a)?,
            upper_a: stack_matrix_cache_rows(caches, |cache| &cache.upper_a)?,
            lower_b: stack_vector_cache_rows(caches, |cache| &cache.lower_b)?,
            upper_b: stack_vector_cache_rows(caches, |cache| &cache.upper_b)?,
        })
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
}
