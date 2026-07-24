// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shape compatibility utilities for bound propagation.
//!
//! Pure shape-computation helpers (no dependency on `network/` or `layers/`).
//! Used by `layers/`, `bounds/`, and `ny-build` for broadcasting checks.

/// Compute the broadcast shape for two shapes according to NumPy/ONNX broadcasting rules.
///
/// Returns None if the shapes are not broadcastable.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);

    // Pad shorter shape with 1s from the left
    let a_padded: Vec<usize> = std::iter::repeat_n(1, max_len - a.len())
        .chain(a.iter().copied())
        .collect();
    let b_padded: Vec<usize> = std::iter::repeat_n(1, max_len - b.len())
        .chain(b.iter().copied())
        .collect();

    for (da, db) in a_padded.iter().zip(b_padded.iter()) {
        if *da == *db {
            result.push(*da);
        } else if *da == 1 {
            result.push(*db);
        } else if *db == 1 {
            result.push(*da);
        } else {
            return None; // Not broadcastable
        }
    }

    Some(result)
}

/// Build a flat-index mapping from broadcast output to a (possibly smaller) input.
///
/// For each flat index `j` in the broadcast output (row-major), computes
/// the corresponding flat index in `input_shape`. When `input_shape` has a
/// dimension of size 1 that was broadcast to match `output_shape`, all output
/// positions along that axis map back to the single input element.
///
/// # Example — SE block `[512, 1]` broadcast to `[512, 5]`
/// ```text
/// output_shape = [512, 5], input_shape = [512, 1]
/// output[i*5 + j] uses input[i]  (j ∈ 0..5 all map to same i)
/// ```
///
/// Reference: alpha-beta-CROWN `reduce_broadcast_dims` (utils.py:408-449).
pub fn broadcast_flat_index_map(output_shape: &[usize], input_shape: &[usize]) -> Vec<usize> {
    let n: usize = output_shape.iter().product();
    let ndim = output_shape.len();
    let input_ndim = input_shape.len();
    let offset = ndim.saturating_sub(input_ndim);

    // Precompute input strides (row-major)
    let mut input_strides = vec![1usize; input_ndim];
    for d in (0..input_ndim.saturating_sub(1)).rev() {
        input_strides[d] = input_strides[d + 1] * input_shape[d + 1];
    }

    let mut map = Vec::with_capacity(n);
    for flat_idx in 0..n {
        let mut remainder = flat_idx;
        let mut input_flat = 0usize;
        for d in (0..ndim).rev() {
            let out_idx = remainder % output_shape[d];
            remainder /= output_shape[d];
            if d >= offset {
                let input_d = d - offset;
                let in_idx = if input_shape[input_d] == 1 {
                    0
                } else {
                    out_idx
                };
                input_flat += in_idx * input_strides[input_d];
            }
        }
        map.push(input_flat);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ntest::timeout(5000)]
    #[test]
    fn test_broadcast_shapes_identical() {
        assert_eq!(
            broadcast_shapes(&[2, 3, 4], &[2, 3, 4]),
            Some(vec![2, 3, 4])
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_broadcast_shapes_scalar() {
        assert_eq!(broadcast_shapes(&[], &[3, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[3, 4], &[]), Some(vec![3, 4]));
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_broadcast_shapes_ones() {
        assert_eq!(broadcast_shapes(&[1, 4], &[3, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[3, 1], &[3, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[1, 1], &[3, 4]), Some(vec![3, 4]));
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_broadcast_shapes_different_ndim() {
        assert_eq!(broadcast_shapes(&[4], &[3, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[2, 3, 4], &[4]), Some(vec![2, 3, 4]));
        assert_eq!(
            broadcast_shapes(&[1, 3, 1], &[2, 1, 4]),
            Some(vec![2, 3, 4])
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_broadcast_shapes_incompatible() {
        assert_eq!(broadcast_shapes(&[3], &[4]), None);
        assert_eq!(broadcast_shapes(&[2, 3], &[4, 3]), None);
        assert_eq!(broadcast_shapes(&[2, 3, 4], &[2, 5, 4]), None);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_broadcast_shapes_batch_broadcast_pattern() {
        assert_eq!(
            broadcast_shapes(&[32, 1, 64], &[1, 8, 64]),
            Some(vec![32, 8, 64])
        );
        assert_eq!(
            broadcast_shapes(&[32, 8, 1], &[1, 1, 64]),
            Some(vec![32, 8, 64])
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_broadcast_flat_index_map_identity() {
        // No broadcast: identity mapping
        let map = broadcast_flat_index_map(&[3, 4], &[3, 4]);
        assert_eq!(map, (0..12).collect::<Vec<_>>());
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_broadcast_flat_index_map_se_block_pattern() {
        // SE block: [512, 1] broadcast to [512, 5]
        // output[i*5 + j] maps to input[i] for all j
        let map = broadcast_flat_index_map(&[4, 3], &[4, 1]);
        // output shape [4, 3] = 12 elements, input shape [4, 1] = 4 elements
        // output[0,0]=0 -> input[0], output[0,1]=1 -> input[0], output[0,2]=2 -> input[0]
        // output[1,0]=3 -> input[1], etc.
        assert_eq!(map, vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_broadcast_flat_index_map_leading_broadcast() {
        // [1, 4] broadcast to [3, 4]
        // output[i*4 + j] maps to input[j]
        let map = broadcast_flat_index_map(&[3, 4], &[1, 4]);
        assert_eq!(map, vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_broadcast_flat_index_map_scalar_broadcast() {
        // [1] broadcast to [3, 2]
        let map = broadcast_flat_index_map(&[3, 2], &[1]);
        assert_eq!(map, vec![0, 0, 0, 0, 0, 0]);
    }
}
