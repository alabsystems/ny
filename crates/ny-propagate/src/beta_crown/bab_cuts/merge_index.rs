// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Normalize a coefficient to its sign key for pass-local merge indexing.
pub(super) fn sign_i8(coeff: f32) -> i8 {
    if coeff > 0.0 {
        1
    } else if coeff < 0.0 {
        -1
    } else {
        0
    }
}

/// Mark which prepared signatures should be kept after first-seen deduplication.
pub(super) fn dedup_signatures<S, I>(signatures: I) -> Vec<bool>
where
    I: IntoIterator<Item = S>,
    S: Eq + Hash,
{
    let mut seen = HashSet::new();
    let mut keep = Vec::new();

    for signature in signatures {
        keep.push(seen.insert(signature));
    }

    keep
}

/// Pass-local index for finding previously accepted parent signatures that are
/// subsets of a candidate child signature.
#[derive(Debug)]
pub(super) struct ParentSubsetIndex<K>
where
    K: Copy + Eq + Hash,
{
    accepted_signatures: Vec<Vec<K>>,
    postings: HashMap<K, Vec<usize>>,
}

impl<K> ParentSubsetIndex<K>
where
    K: Copy + Eq + Hash,
{
    pub(super) fn new() -> Self {
        Self {
            accepted_signatures: Vec::new(),
            postings: HashMap::new(),
        }
    }

    pub(super) fn insert(&mut self, signature: &[K]) {
        if signature.is_empty() {
            return;
        }

        let parent_id = self.accepted_signatures.len();
        self.accepted_signatures.push(signature.to_vec());

        for &key in signature {
            self.postings.entry(key).or_default().push(parent_id);
        }
    }

    pub(super) fn has_parent_subset(&self, child_signature: &[K]) -> bool {
        if child_signature.is_empty() || self.accepted_signatures.is_empty() {
            return false;
        }

        let child_keys: HashSet<K> = child_signature.iter().copied().collect();
        if child_keys.is_empty() {
            return false;
        }

        let mut posting_lists: Vec<&[usize]> = child_keys
            .iter()
            .filter_map(|key| self.postings.get(key).map(|ids| ids.as_slice()))
            .collect();
        posting_lists.sort_by_key(|ids| ids.len());

        let mut seen_candidate_ids = HashSet::with_capacity(self.accepted_signatures.len());

        for candidate_ids in posting_lists {
            for &parent_id in candidate_ids {
                if !seen_candidate_ids.insert(parent_id) {
                    continue;
                }

                let parent_signature = &self.accepted_signatures[parent_id];
                if parent_signature.len() >= child_signature.len() {
                    continue;
                }

                if parent_signature.iter().all(|key| child_keys.contains(key)) {
                    return true;
                }
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::ParentSubsetIndex;

    #[test]
    fn finds_subset_parent_outside_smallest_posting_list() {
        let mut index = ParentSubsetIndex::new();
        index.insert(&[1]);
        index.insert(&[1, 4]);
        index.insert(&[1, 5]);
        index.insert(&[2, 6]);

        assert!(index.has_parent_subset(&[1, 2, 3]));
    }

    #[test]
    fn rejects_non_subset_candidates() {
        let mut index = ParentSubsetIndex::new();
        index.insert(&[1, 4]);
        index.insert(&[2, 5]);

        assert!(!index.has_parent_subset(&[1, 2, 3]));
    }
}
