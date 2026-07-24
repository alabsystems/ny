// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Fast lookup iterators for [`BetaState`] entries.
//!
//! Extracted from `beta.rs` to stay within the 500-line file limit.

use super::BetaEntry;

/// Iterator over [`BetaEntry`] references for a specific layer.
///
/// When lookup indexes are fresh, uses pre-built index for O(k) iteration
/// (k = entries for this layer). Falls back to linear scan otherwise.
pub(crate) enum BetaEntriesForLayer<'a> {
    Indexed {
        indices: std::slice::Iter<'a, usize>,
        entries: &'a [BetaEntry],
    },
    Linear {
        entries: std::slice::Iter<'a, BetaEntry>,
        layer_idx: usize,
    },
}

impl<'a> Iterator for BetaEntriesForLayer<'a> {
    type Item = &'a BetaEntry;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Indexed { indices, entries } => indices.next().map(|&idx| &entries[idx]),
            Self::Linear { entries, layer_idx } => {
                entries.find(|entry| entry.layer_idx == *layer_idx)
            }
        }
    }
}
