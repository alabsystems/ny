// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Internal reshape sentinels for dimensions copied from a specific input axis.

const COPY_AXIS_SENTINEL_BASE: i64 = i64::MIN / 2;
const COPY_AXIS_SENTINEL_LIMIT: usize = 4096;

/// Encode "copy input dimension at `axis`" for internal reshape targets.
///
/// ONNX's public `0` reshape sentinel only copies the dimension at the same
/// target index. Shape/Gather/Concat chains can move a runtime dimension to a
/// different target index, so the loader uses this private sentinel instead.
#[must_use]
pub fn reshape_copy_axis_sentinel(axis: usize) -> Option<i64> {
    if axis >= COPY_AXIS_SENTINEL_LIMIT {
        return None;
    }
    Some(COPY_AXIS_SENTINEL_BASE + axis as i64)
}

/// Decode an internal reshape copy-axis sentinel.
#[must_use]
pub fn reshape_copy_axis_from_sentinel(dim: i64) -> Option<usize> {
    let offset = dim.checked_sub(COPY_AXIS_SENTINEL_BASE)?;
    if offset < 0 || offset >= COPY_AXIS_SENTINEL_LIMIT as i64 {
        return None;
    }
    usize::try_from(offset).ok()
}

#[cfg(test)]
mod tests {
    use super::{reshape_copy_axis_from_sentinel, reshape_copy_axis_sentinel};

    #[test]
    fn copy_axis_sentinel_round_trips() {
        for axis in [0, 1, 17, 4095] {
            let sentinel = reshape_copy_axis_sentinel(axis).expect("axis in range");
            assert_eq!(reshape_copy_axis_from_sentinel(sentinel), Some(axis));
        }
    }

    #[test]
    fn ordinary_negative_dims_are_not_copy_axis_sentinels() {
        for dim in [-1, -2, -100] {
            assert_eq!(reshape_copy_axis_from_sentinel(dim), None);
        }
    }
}
